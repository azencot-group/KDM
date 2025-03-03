from tqdm import tqdm
import torch
import numpy as np


def log_losses(epoch, losses_tr, losses_te, names):
    losses_avg_tr, losses_avg_te = [], []

    for loss in losses_tr:
        losses_avg_tr.append(np.mean(loss))

    for loss in losses_te:
        losses_avg_te.append(np.mean(loss))

    loss_str_tr = 'Epoch {}, TRAIN: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_tr):
        loss_str_tr += '{}={:.3e}, \t'.format(names[jj], loss)
    print(loss_str_tr)

    loss_str_te = 'Epoch {}, TEST: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_te):
        loss_str_te += '{}={:.3e}, \t'.format(names[jj], loss)
    print(loss_str_te)

    return losses_avg_tr[0], losses_avg_te[0]


def agg_losses(LOSSES, losses):
    if not LOSSES:
        LOSSES = [[] for _ in range(len(losses))]
    for jj, loss in enumerate(losses):
        LOSSES[jj].append(loss.item())
    return LOSSES


def train(model, optimizer, train_loader, test_loader, epochs=50):
    print(f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for epoch in range(0, epochs):
        print("Running Epoch : {}".format(epoch + 1))

        model.train()
        losses_agg_tr, losses_agg_te = [], []
        for i, data in tqdm(enumerate(train_loader, 1)):
            X = data.cuda().float()

            optimizer.zero_grad()
            outputs = model(X)

            losses = model.loss(X, outputs)
            losses[0].backward()
            optimizer.step()

            losses_agg_tr = agg_losses(losses_agg_tr, losses)

        model.eval()
        with torch.no_grad():
            print('Evaulating the model')
            for i, data in tqdm(enumerate(test_loader, 1)):
                X = data.cuda().float()

                outputs = model(X)
                losses = model.loss(X, outputs)

                losses_agg_te = agg_losses(losses_agg_te, losses)

        # log losses
        log_losses(epoch, losses_agg_tr, losses_agg_te, model.names)

    print("Training is complete")
