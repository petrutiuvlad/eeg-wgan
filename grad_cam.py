import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
sns.set()


class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        preds, features, gradients = self.model(input)
        prediction = preds

        if index is None:
            index = np.argmax(preds.cpu().data.numpy())

        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, preds.size()[-1]).zero_()
        one_hot_output[0][index] = 1
        # # Zero grads
        self.model.zero_grad()
        # #Backwards pass with specified target
        preds.backward(gradient=one_hot_output, retain_graph=True)
        # #Get hooked gradients
        guided_gradients = gradients.cpu().data.numpy()[0]
        # #Get conv outputs
        target = features.cpu().data.numpy()[0, :]

        # Get weights from gradients
        # Take average for each gradient
        weights = np.mean(guided_gradients, axis=1)
        #print("weights {}".format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # print("cam {}".format(cam.shape))

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        cam_i = np.maximum(cam, 0)

        cam_i = cam_i - np.min(cam_i)
        cam_i = cam_i / np.max(cam_i)

        vec = torch.from_numpy(cam_i)
        vec = vec.reshape(1, 1, 127)
        new_vec = torch.nn.functional.interpolate(vec,
                                                  size=(512),
                                                  mode='linear',
                                                  align_corners=True)
        return new_vec, prediction


def plot_gradcam(path_channel,
                 model,
                 dataset,
                 train_dataset=False,
                 generated=False):
    mean_importance_unseen = torch.FloatTensor(512).zero_()
    mean_importance_seen = torch.FloatTensor(512).zero_()
    number_seen = 0
    number_unseen = 0
    grad_cam = GradCam(model=model)
    if train_dataset == True:
        path_channel += 'train/'
    else:
        path_channel += 'test/'

    if generated == True:
        path_channel += 'gan/'
    else:
        path_channel += 'raw/'

    for i in range(0, len(dataset)):
        importance, prediction = grad_cam(dataset[i][0].cuda(), None)
        # plt.rcParams["figure.figsize"] = (20, 7)
        # fig, ((ax1, ax2)) = plt.subplots(2, 1)
        # ax1.plot(dataset[i][0])
        importance = importance.view(512)
        importance[importance != importance] = 0
        # ax2.plot(importance)

        if (dataset[i][1] == 0):
            if (torch.round(prediction.detach()) == dataset[i][1]):
                # plt.savefig(path_channel +
                #             'gan/Predictions/Unseen/Good/{}.png'.format(i))
                # plt.close(fig)
                mean_importance_unseen = mean_importance_unseen.add(importance)
                number_unseen += 1
            # else:
            #     plt.savefig(path_channel +
            #                 'gan/Predictions/Unseen/Wrong/{}.png'.format(i))
            #     plt.close(fig)
        else:
            if (torch.round(prediction.detach()) == dataset[i][1]):
                # plt.savefig(path_channel +
                #             'gan/Predictions/Seen/Good/{}.png'.format(i))
                # plt.close(fig)
                mean_importance_seen = mean_importance_seen.add(importance)
                number_seen += 1
            # else:
            #     plt.savefig(path_channel +
            #                 'gan/Predictions/Seen/Wrong/{}.png'.format(i))
            #     plt.close(fig)

    mean_importance_seen /= number_seen
    mean_importance_unseen /= number_unseen

    plt.rcParams["figure.figsize"] = (20, 7)
    sns.lineplot(data=mean_importance_seen)
    plt.savefig(path_channel + 'Predictions/Seen/mean.png')
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (20, 7)
    sns.lineplot(data=mean_importance_unseen)
    plt.savefig(path_channel + 'Predictions/Unseen/mean.png')
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (20, 7)
    sns.lineplot(data=(mean_importance_seen - mean_importance_unseen))
    plt.savefig(path_channel + 'Predictions/mean.png')
    plt.clf()
    plt.close()

    f = open(path_channel + "mean_importance_seen.pkl", "wb")
    pickle.dump(mean_importance_seen, f)
    f.close()
    f = open(path_channel + "mean_importance_unseen.pkl", "wb")
    pickle.dump(mean_importance_unseen, f)
    f.close()

    if generated == True:
        if train_dataset == True:
            mean_importance_unseen_raw = torch.FloatTensor(512).zero_()
            mean_importance_seen_raw = torch.FloatTensor(512).zero_()
            number_seen_raw = 0
            number_unseen_raw = 0

            for i in range(0, len(dataset) - 400):
                importance, prediction = grad_cam(dataset[i][0].cuda(), None)
                importance = importance.view(512)
                importance[importance != importance] = 0

                if (dataset[i][1] == 0):
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_unseen_raw = mean_importance_unseen_raw.add(
                            importance)
                        number_unseen_raw += 1
                else:
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_seen_raw = mean_importance_seen_raw.add(
                            importance)
                        number_seen_raw += 1

            mean_importance_seen_raw /= number_seen_raw
            mean_importance_unseen_raw /= number_unseen_raw

            mean_importance_unseen_gen = torch.FloatTensor(512).zero_()
            mean_importance_seen_gen = torch.FloatTensor(512).zero_()
            number_seen_gen = 0
            number_unseen_gen = 0

            for i in range(len(dataset) - 400, len(dataset)):
                importance, prediction = grad_cam(dataset[i][0].cuda(), None)
                importance = importance.view(512)
                importance[importance != importance] = 0

                if (dataset[i][1] == 0):
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_unseen_gen = mean_importance_unseen_gen.add(
                            importance)
                        number_unseen_gen += 1
                else:
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_seen_gen = mean_importance_seen_gen.add(
                            importance)
                        number_seen_gen += 1

            mean_importance_seen_gen /= number_seen_gen
            mean_importance_unseen_gen /= number_unseen_gen

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_seen_raw)
            plt.savefig(path_channel + 'Predictions/Seen/mean_raw.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_seen_gen)
            plt.savefig(path_channel + 'Predictions/Seen/mean_gen.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_unseen_raw)
            plt.savefig(path_channel + 'Predictions/Unseen/mean_raw.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_unseen_gen)
            plt.savefig(path_channel + 'Predictions/Unseen/mean_gen.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_unseen_raw, label='RAW')
            sns.lineplot(data=mean_importance_unseen_gen, label='GEN')
            plt.legend('upper right')
            plt.savefig(path_channel + 'Predictions/Unseen/mean_raw_gen.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_seen_raw, label='RAW')
            sns.lineplot(data=mean_importance_seen_gen, label='GEN')
            plt.legend('upper right')
            plt.savefig(path_channel + 'Predictions/Seen/mean_raw_gen.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=(mean_importance_seen_raw -
                               mean_importance_unseen_raw))
            plt.savefig(path_channel + 'Predictions/mean_raw.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=(mean_importance_seen_gen -
                               mean_importance_unseen_gen))
            plt.savefig(path_channel + 'Predictions/mean_gen.png')
            plt.clf()
            plt.close()

            f = open(path_channel + "mean_importance_seen_gen.pkl", "wb")
            pickle.dump(mean_importance_seen_gen, f)
            f.close()
            f = open(path_channel + "mean_importance_seen_raw.pkl", "wb")
            pickle.dump(mean_importance_seen_raw, f)
            f.close()
            f = open(path_channel + "mean_importance_unseen_gen.pkl", "wb")
            pickle.dump(mean_importance_unseen_gen, f)
            f.close()
            f = open(path_channel + "mean_importance_unseen_raw.pkl", "wb")
            pickle.dump(mean_importance_unseen_raw, f)
            f.close()
    else:
        if train_dataset == True:
            mean_importance_unseen_raw = torch.FloatTensor(512).zero_()
            mean_importance_seen_raw = torch.FloatTensor(512).zero_()
            number_seen_raw = 0
            number_unseen_raw = 0

            for i in range(0, len(dataset)):
                importance, prediction = grad_cam(dataset[i][0].cuda(), None)
                importance = importance.view(512)
                importance[importance != importance] = 0

                if (dataset[i][1] == 0):
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_unseen_raw = mean_importance_unseen_raw.add(
                            importance)
                        number_unseen_raw += 1
                else:
                    if (torch.round(prediction.detach()) == dataset[i][1]):
                        mean_importance_seen_raw = mean_importance_seen_raw.add(
                            importance)
                        number_seen_raw += 1

            mean_importance_seen_raw /= number_seen_raw
            mean_importance_unseen_raw /= number_unseen_raw

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_seen_raw)
            plt.savefig(path_channel + 'Predictions/Seen/mean_raw.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=mean_importance_unseen_raw)
            plt.savefig(path_channel + 'Predictions/Unseen/mean_raw.png')
            plt.clf()
            plt.close()

            plt.rcParams["figure.figsize"] = (20, 7)
            sns.lineplot(data=(mean_importance_seen_raw -
                               mean_importance_unseen_raw))
            plt.savefig(path_channel + 'Predictions/mean_raw.png')
            plt.clf()
            plt.close()

            f = open(path_channel + "mean_importance_seen_raw.pkl", "wb")
            pickle.dump(mean_importance_seen_raw, f)
            f.close()
            f = open(path_channel + "mean_importance_unseen_raw.pkl", "wb")
            pickle.dump(mean_importance_unseen_raw, f)
            f.close()


def create_gradcam_dirs(path_channel):
    os.makedirs(path_channel + 'train', exist_ok=True)
    os.makedirs(path_channel + 'test', exist_ok=True)
    os.makedirs(path_channel + 'train/gan', exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions', exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Seen', exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Seen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Seen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Unseen', exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Unseen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'train/gan/Predictions/Unseen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'test/gan', exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions', exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Seen', exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Seen/Good', exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Seen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Unseen', exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Unseen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'test/gan/Predictions/Unseen/Wrong',
                exist_ok=True)

    os.makedirs(path_channel + 'train/raw', exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions', exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Seen', exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Seen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Seen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Unseen', exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Unseen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'train/raw/Predictions/Unseen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'test/raw', exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions', exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Seen', exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Seen/Good', exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Seen/Wrong',
                exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Unseen', exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Unseen/Good',
                exist_ok=True)
    os.makedirs(path_channel + 'test/raw/Predictions/Unseen/Wrong',
                exist_ok=True)
