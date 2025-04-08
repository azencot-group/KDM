# code was inspired by the code repository of flow matching
import copy

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (destroy_process_group,barrier)

from tqdm.auto import tqdm, trange
from accelerate.utils import set_seed

from koopman_distillation.evaluation.fid import sample_and_calculate_fid
from koopman_distillation.evaluation.wassertien_distance import measure_wess_distance
from koopman_distillation.model.modules.model_cifar10 import Discriminator
from koopman_distillation.utils.loggers.logging import plot_samples
from koopman_distillation.utils.dist_lib import get_rank, get_world_size, is_distributed, is_main_process, ddp_sync, \
    gather_logs
from old.distillation.utils.display import plot_spectrum


class MockDDPKoopman(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.module = model

    def forward(self, xt, xT, labels=None):
        return self.module(xt, xT, labels)


class MockDDPDiscriminator(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.module = model

    def forward(self, x):
        return self.module(x)


class TrainLoop:
    def __init__(self, args, model, train_data, test_data, batch_size, device, logger,
                 num_accumulation_rounds, iterations=400001, lr=0.0003, print_every=50, data_shape=(2),
                 teach_model=False, advers=False, cond=False):
        self.args = args
        self.model = model
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.num_accumulation_rounds = num_accumulation_rounds
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.iterations = iterations
        self.print_every = print_every
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.logger = logger
        self.TModel_exists = teach_model
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.best_fid = float('inf')

        set_seed(self.args.seed + get_rank())

        self.discriminator = None
        self.advers = advers
        self.cond = cond
        if self.advers:
            self.discriminator = Discriminator(in_channels=3).to(self.device)
            self.optimizer_adv = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999),
                                                  eps=1e-8)

        if is_distributed(self.args):
            print("Distributed training")
            self.model = DDP(self.model, device_ids=[device])
            self.discriminator = DDP(self.discriminator, device_ids=[device])
        else:
            self.model = MockDDPKoopman(self.model)
            self.discriminator = MockDDPDiscriminator(self.discriminator)
        if is_main_process(self.args):
            print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train(self):
        to_log = {}

        if is_distributed(self.args):
            barrier()

        self.model.to(self.device)

        i = 0
        pbar = tqdm(total=self.iterations, desc=f"Step #", disable=(get_rank() != 0), initial=i)
        while i < self.iterations:

            self.optimizer.zero_grad()
            if self.advers:
                self.optimizer_adv.zero_grad()

            for round_idx in range(self.num_accumulation_rounds):
                with ddp_sync(self.model, (round_idx == self.num_accumulation_rounds - 1)):
                    batch = next(self.train_data)
                    xt, xT, labels = batch
                    if torch.isnan(labels).any():
                        labels = None
                    else:
                        labels = labels.to(self.device)
                    xt = xt.to(self.device)
                    xT = xT.to(self.device)

                    fw_comp = self.model(xt, xT, labels)

                    if self.advers:
                        advers_loss = self.discriminator.module.loss(fw_comp)
                        advers_loss['adv_loss'].backward()
                        for key, value in advers_loss.items():
                            if key not in to_log:
                                to_log[key] = 0
                            to_log[key] += value

                    # calculate loss
                    losses = self.model.module.loss(fw_comp, discriminator=self.discriminator)

                    for key, value in losses.items():
                        if key not in to_log:
                            to_log[key] = 0
                        to_log[key] += value

                    losses['loss'].backward()  # backward

            if self.advers:
                self.optimizer_adv.step()

            self._nan_to_num(self.model.module)
            self.optimizer.step()  # update
            self._update_ema(self.model.module, self.ema)

            if self.advers:
                losses.update(advers_loss)

            if is_distributed(self.args) and i > 0 and (i + 1) % self.print_every == 0:
                to_log = gather_logs(self.args, to_log)
            for key in to_log.keys():
                if is_main_process(self.args) and i > 0 and (i + 1) % self.print_every == 0:
                    self.logger.log(f'train/{key}',
                                    to_log[key] / get_world_size() / self.num_accumulation_rounds, i)
                to_log[key] = 0

            if (i + 1) % self.print_every == 0:

                if is_main_process(self.args):
                    # log the losses
                    self.model.eval()
                    self.evaluation_of_train_and_generation(i + 1)
                    self.model.train()

                barrier()

            i += 1
            pbar.update()

        self.model.eval()
        # self.evaluation_of_test_data(i + 1)
        self.evaluation_of_train_and_generation(i + 1)

        barrier()
        # Cleanup
        destroy_process_group()

    def evaluation_of_train_and_generation(self, iteration):

        # evaluate fid for cifar10
        if iteration % (self.print_every * 100) == 0 and self.data_shape[0] == 3:

            plot_spectrum(self.model.module.koopman_operator.weight.data.cpu().detach().numpy(), self.args.workdir,
                          self.logger)

            # plot qualitative results
            plot_samples(self.logger, self.ema, self.batch_size, self.device, self.data_shape, self.args.workdir,
                         self.cond)

            fid_ema = sample_and_calculate_fid(model=self.ema,
                                               data_shape=self.data_shape,
                                               num_samples=50_000,
                                               device=self.device,
                                               batch_size=self.batch_size,
                                               epoch=iteration,
                                               image_dir=self.args.workdir,
                                               cond=self.cond,
                                               )
            self.logger.log('ema_fid', fid_ema, iteration)
            fid_model = sample_and_calculate_fid(model=self.model.module,
                                                 data_shape=self.data_shape,
                                                 num_samples=50_000,
                                                 device=self.device,
                                                 batch_size=self.batch_size,
                                                 epoch=iteration,
                                                 image_dir=self.args.workdir,
                                                 cond=self.cond,
                                                 )
            self.logger.log('model_fid', fid_model, iteration)

            if min(fid_model, fid_ema) < self.best_fid:
                if is_distributed(self.args):
                    checkpoint = {
                        "model": self.model.module.state_dict(),
                        "ema": self.ema.state_dict(),
                        "opt": self.optimizer.state_dict(),
                        "args": self.args,
                        "steps": iteration,
                    }
                    checkpoint_path = f"{self.args.workdir}/{iteration:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                else:
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "ema": self.ema.state_dict(),
                        "opt": self.optimizer.state_dict(),
                        "args": self.args,
                        "steps": iteration,
                    }
                    checkpoint_path = f"{self.args.workdir}/{iteration:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                self.best_fid = min(fid_model, fid_ema)

            # save the model

        # # checkerboard evaluation
        # elif iteration % (self.print_every * 10) == 0 and self.data_shape[0] == 2:
        #     torch.save(self.model, f'{self.output_dir}/model.pt')
        #     wess_distance = measure_wess_distance(self.model, self.device, self.train_data, num_samples=40000)
        #     self.logger.log('wess_distance', wess_distance, iteration)
        #     # save the model

    def evaluation_of_test_data(self, iteration):
        if self.test_data is None or self.cond:
            return

        loss_sums = {}
        num_batches = 0

        for test_batch in self.test_data:
            xT, xt, labels = test_batch
            if torch.isnan(labels).any():
                labels = None
            else:
                labels = labels.to(self.device)
            xt = xt.to(self.device)
            xT = xT.to(self.device)
            # return all components relevant for loss calculation
            fw_comp = self.model(x_0=xt, x_T=xT, labels=labels)
            # calculate loss
            test_losses = self.model.module.loss(fw_comp, self.discriminator)

            for k, v in test_losses.items():
                if k not in loss_sums:
                    loss_sums[k] = 0.0
                loss_sums[k] += v.item()

            num_batches += 1

            # report losses
        for k, total_loss in loss_sums.items():
            avg_loss = total_loss / num_batches
            self.logger.log(f'test/{k}', avg_loss, iteration)

    def _nan_to_num(self, net):
        # Update weights.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

    def _update_ema(self, net, ema, ema_beta=0.9999):
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
