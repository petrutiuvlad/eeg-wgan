import matplotlib.pyplot as plt
from train_classifier import train
import generate_data as data
import os

for i in range(1, 129):
    data.generate_signals(i)
    train_losses, train_accuracies, test_losses, test_accuracies = train()
    os.makedirs("./data/results_classify/channel_{}".format(i),
                exist_ok=True)  # succeeds even if directory exists.

    plt.plot(train_losses, 'b', label='Train Loss')
    plt.plot(test_losses, 'r', label='Test Loss')
    plt.legend(loc="upper left")
    plt.savefig('./data/results_classify/channel_{}/losses.png'.format(i))
    plt.clf()

    plt.plot(train_accuracies, 'b', label='Train Acc')
    plt.plot(test_accuracies, 'r', label='Test Acc')
    plt.legend(loc="upper left")
    plt.savefig('./data/results_classify/channel_{}/accuracies.png'.format(i))
    plt.clf()

    print('--------GENERATED CHANNEL{}------------'.format(i))
