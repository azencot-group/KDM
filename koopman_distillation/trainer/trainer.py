# code was inspired by the code repository of flow matching
import time
from matplotlib import pyplot as plt

import torch

from koopman_distillation.evaluation.fid import sample_and_calculate_fid
from koopman_distillation.utils.loggers.logging import plot_samples


class TrainLoop:
    def __init__(self, model, train_data, batch_size, device, output_dir, logger, iterations=1001, lr=0.0003,
                 print_every=50, data_shape=(2)):
        self.model = model
        self.train_data = train_data
        self.device = device
        self.iterations = iterations
        self.print_every = print_every
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.logger = logger

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self):
        start_time = time.time()
        for i in range(self.iterations):
            for batch in self.train_data:
                # todo = models should get the path and decide what to do with it
                xt, xT, _ = batch
                xt = xt.to(self.device)
                xT = xT.to(self.device)

                self.optimizer.zero_grad()

                # return all components relevant for loss calculation
                fw_comp = self.model(x_0=xt, x_T=xT)

                # calculate loss
                losses = self.model.loss(fw_comp)

                losses['loss'].backward()  # backward
                self.optimizer.step()  # update

            if (i + 1) % self.print_every == 0:
                self.model.eval()
                self.evaluation(start_time, losses, i + 1)
                start_time = time.time()
                self.model.train()

    def evaluation(self, start_time, losses, iteration):
        # log the losses
        elapsed = time.time() - start_time
        self.logger.log(f'test/elapsed', elapsed * 1000 / self.print_every, iteration)
        for k, v in losses.items():
            self.logger.log(k, v.item(), iteration)

        # plot qualitative results
        plot_samples(self.logger, self.model, self.batch_size, self.device, self.data_shape, self.output_dir, iteration)

        # evaluate fid
        if iteration % (self.print_every * 10) == 0:
            fid = sample_and_calculate_fid(model=self.model,
                                           data_shape=self.data_shape,
                                           num_samples=50000,
                                           device=self.device,
                                           batch_size=self.batch_size,
                                           epoch=iteration,
                                           image_dir=self.output_dir)
            self.logger.log('fid', fid, iteration)
