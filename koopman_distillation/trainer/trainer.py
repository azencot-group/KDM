# code was inspired by the code repository of flow matching
import time
import copy

import numpy as np
import torch

from koopman_distillation.evaluation.fid import sample_and_calculate_fid
from koopman_distillation.evaluation.wassertien_distance import measure_wess_distance
from koopman_distillation.model.modules.model_cifar10 import Discriminator
from koopman_distillation.utils.loggers.logging import plot_samples
from old.distillation.utils.display import plot_spectrum


class TrainLoop:
    def __init__(self, model, train_data, test_data, batch_size, device, output_dir, logger, ema_rate,
                 iterations=400001, lr=0.0003, print_every=50, data_shape=(2), teach_model=False, advers=False,
                 cond=False):
        self.model = model
        # self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.train_data = iter(train_data)
        self.test_data = test_data
        self.device = device
        self.iterations = iterations
        self.print_every = print_every
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.logger = logger
        self.TModel_exists = teach_model
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        # self.best_fid_ema = float('inf')
        self.best_fid_model = float('inf')

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.discriminator = None
        self.advers = advers
        self.cond = cond
        if advers:
            self.discriminator = Discriminator(in_channels=3).cuda()
            self.optimizer_adv = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999),
                                                  eps=1e-8)

    def train(self):
        start_time = time.time()
        i = 0
        while i < self.iterations:
            batch = next(self.train_data)
            xt, xT, labels = batch
            if torch.isnan(labels).any():
                labels = None
            else:
                labels = labels.to(self.device)
            xt = xt.to(self.device)
            xT = xT.to(self.device)

            self.optimizer.zero_grad()

            # return all components relevant for loss calculation
            fw_comp = self.model(x_0=xt, x_T=xT, labels=labels)

            # --- disc losses --- #
            if self.advers:
                self.optimizer_adv.zero_grad()
                advers_loss = self.discriminator.loss(fw_comp)
                advers_loss['adv_loss'].backward()
                self.optimizer_adv.step()

            # --- koopman losses --- #
            losses = self.model.loss(fw_comp, discriminator=self.discriminator)
            losses['loss'].backward()  # backward
            self._nan_to_num(self.model)
            self.optimizer.step()  # update
            # self._update_ema(self.model, self.ema)

            if self.advers:
                losses.update(advers_loss)

            # --- evaluations --- #
            if (i + 1) % self.print_every == 0:
                self.model.eval()
                self.evaluation_of_test_data(i + 1)
                self.evaluation_of_train_and_generation(start_time, losses, i + 1)
                start_time = time.time()
                self.model.train()

            i += 1

    def evaluation_of_train_and_generation(self, start_time, losses, iteration):
        # log the losses
        elapsed = time.time() - start_time
        self.logger.log(f'test/elapsed', elapsed * 1000 / self.print_every, iteration)
        for k, v in losses.items():
            self.logger.log(k, v.item(), iteration)


        # evaluate fid for cifar10
        if iteration % (self.print_every * 100) == 0 and self.data_shape[0] == 3:
            # fid_ema = sample_and_calculate_fid(model=self.ema,
            #                                    data_shape=self.data_shape,
            #                                    num_samples=50000,
            #                                    device=self.device,
            #                                    batch_size=self.batch_size,
            #                                    epoch=iteration,
            #                                    image_dir=self.output_dir,
            #                                    cond=self.cond,
            #                                    )
            # self.logger.log('ema_fid', fid_ema, iteration)
            fid_model = sample_and_calculate_fid(model=self.model,
                                                 data_shape=self.data_shape,
                                                 num_samples=50000,
                                                 device=self.device,
                                                 batch_size=self.batch_size,
                                                 epoch=iteration,
                                                 image_dir=self.output_dir,
                                                 cond=self.cond,
                                                 )
            self.logger.log('model_fid', fid_model, iteration)
            plot_spectrum(self.model.koopman_operator.weight.data.cpu().detach().numpy(), self.output_dir, self.logger)
            # plot qualitative results
            plot_samples(self.logger, self.ema, self.batch_size, self.device, self.data_shape, self.output_dir,
                         self.cond)

            # if fid_ema < self.best_fid_ema:
            #     torch.save(self.model, f'{self.output_dir}/ema_model.pt')
            #     self.best_fid_ema = fid_ema

            if fid_model < self.best_fid_model:
                torch.save(self.model, f'{self.output_dir}/model.pt')
                self.best_fid_model = fid_model
            # save the model

        # checkerboard evaluation
        elif iteration % (self.print_every * 10) == 0 and self.data_shape[0] == 2:
            torch.save(self.model, f'{self.output_dir}/model.pt')
            wess_distance = measure_wess_distance(self.model, self.device, self.train_data, num_samples=40000)
            self.logger.log('wess_distance', wess_distance, iteration)
            # save the model

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
            test_losses = self.model.loss(fw_comp, self.discriminator)

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
