import torch
import torch.nn as nn
import torch.utils.data as data_utils
from grad_cam import *
from model.model import *
from generate_data.read import *
from evaluation.metrics import *
from utils.utils import *


def train_per_channel_gen(channel: int,
                          path_channel,
                          good_subjects,
                          fold=False,
                          index_test=1,
                          shuffle=False,
                          mode='all'):

    if (fold == True):
        train, test, targets, test_targets = prepare_11_fold(index_test,
                                                             channel,
                                                             good_subjects,
                                                             generated=True)
    elif (shuffle == True):
        train, test, targets, test_targets = prepare_data_shuffle(
            signals_unseen=[],
            signals_seen=[],
            path_channel=path_channel,
            generated=True)
        path_channel += '11Fold/shuffle/'
    else:
        signals_unseen, signals_seen = read_raw_signal(channel, good_subjects)
        train, test, targets, test_targets = prepare_data(signals_unseen,
                                                          signals_seen,
                                                          path_channel,
                                                          generated=True)

    train = transform(train, mode=mode)
    test = transform(test, mode=mode)

    train = data_utils.TensorDataset(train, targets)
    test = data_utils.TensorDataset(test, test_targets)
    loader = data_utils.DataLoader(train, batch_size=16, shuffle=True)
    loader_test = data_utils.DataLoader(test, batch_size=16, shuffle=False)

    discriminator = Linear()
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)
    num_epochs = 5

    loss = nn.BCELoss()

    discriminator = discriminator.cuda()

    with open(path_channel + 'wgan/files/accuracies_{}.txt'.format(mode),
              'w+') as acc_file, open(
                  path_channel + 'wgan/files/losses_{}.txt'.format(mode),
                  'w+') as loss_file:
        for epoch in range(num_epochs):
            train_loss = 0
            train_acc = 0
            discriminator.train()
            for index, (real_batch, targets) in enumerate(loader):
                optimizer_D.zero_grad()

                preds, _, _ = discriminator(real_batch.cuda())

                error = loss(preds, targets.unsqueeze(1).float().cuda())
                acc = binary_acc(preds, targets.unsqueeze(1).float().cuda())
                error.backward()

                optimizer_D.step()
                train_loss += error
                train_acc += acc

            losssss = (train_loss / len(loader))
            acccccc = (train_acc / len(loader))
            loss_file.write('Train:' + str(losssss.item()) + '\n')
            acc_file.write('Train:' + str(acccccc.item()) + '\n')

            val_loss = 0
            val_acc = 0
            discriminator.eval()
            for index, (real_batch, targets) in enumerate(loader_test):
                preds, _, _ = discriminator(real_batch.cuda())

                error = loss(preds, targets.unsqueeze(1).float().cuda())

                acc = binary_acc(preds, targets.unsqueeze(1).float().cuda())

                val_loss += error
                val_acc += acc
            losssss = (val_loss / len(loader_test))
            acccccc = (val_acc / len(loader_test))
            loss_file.write('Test:' + str(losssss.item()) + '\n')
            acc_file.write('Test:' + str(acccccc.item()) + '\n')
            torch.cuda.empty_cache()

        loss_file.close()
        acc_file.close()

    discriminator.eval()
    predictions = []
    ground_truths = []
    for index, (real_batch, targets) in enumerate(loader_test):
        preds, _, _ = discriminator(real_batch.cuda())

        preds_rounded = torch.round(preds.detach())
        predictions.append(preds_rounded.cpu())
        ground_truths.append(targets.detach().cpu())
    torch.save(predictions,
               path_channel + 'wgan/files/preds_{}.pt'.format(mode))
    torch.save(ground_truths,
               path_channel + 'wgan/files/gts_{}.pt'.format(mode))
    plot_gradcam(path_channel,
                 discriminator,
                 test,
                 train_dataset=False,
                 generated=True)
    plot_gradcam(path_channel,
                 discriminator,
                 train,
                 train_dataset=True,
                 generated=True)


# for index in range(1, 129):
#     print('--------------------{}----------------------'.format(index))
#     train_per_channel(index)
# train_per_channel(70)
# for index_test in range(1, 12):
#     train_per_channel_gen(
#         channel=71,
#         path_channel='/home/vlad/Desktop/Statistics_GAN/11_Fold/ch71/{}/'.
#         format(index_test),
#         fold=True,
#         index_test=index_test)

# train_per_channel_gen(
#     channel=71,
#     path_channel='/home/vlad/Desktop/Statistics_GAN/11_Fold/ch71/shuffle/',
#     fold=False,
#     shuffle=True)
