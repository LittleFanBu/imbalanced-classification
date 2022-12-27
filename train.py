import math
import torch
import torchvision
import torchvision.models

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from mydata import *
from AlexNet import *

from torchsampler import ImbalancedDatasetSampler

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((120, 120)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), }

batch_size = 64


def main():
    store_path = './face/train'
    # use my_dataset to load the data, mark the "with_mask" as 0, mark the "without_mask" as 1
    train_data_mask = my_dataset(store_path, 'with_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_data_no_mask = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    #  use transform to do the data-augmentation
    # generate 60 new horizontal flip without_mask data
    train_data_no_mask_hflip = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new vertical flip without_mask data
    train_data_no_mask_fflip = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.RandomVerticalFlip(p=1), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new rotation between 60 and 90 degree without_mask data
    train_data_no_mask_rotate60_90 = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.RandomRotation([60, 90]), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new rotation between 100 and 180 degree without_mask data
    train_data_no_mask_rotate100_180 = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.RandomRotation([100, 180]), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new rotation between 200 and 270 degree without_mask data
    train_data_no_mask_rotate200_270 = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.RandomRotation([200, 270]), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new gaussian blur without_mask data
    train_data_no_mask_gassion_blur = my_dataset(store_path, 'without_mask', transforms.Compose(
        [transforms.RandomResizedCrop(120), transforms.GaussianBlur(3), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # generate 60 new invert without_mask data
    train_data_no_mask_invert = my_dataset(store_path, 'without_mask',
                                           transforms.Compose(
                                               [transforms.RandomResizedCrop(120), transforms.RandomInvert(p=1),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # link these new generate data with the original data
    # 500: 180
    # train_data_cat = ConcatDataset(
    #     [train_data_mask, train_data_no_mask, train_data_no_mask_hflip, train_data_no_mask_fflip])

    # 500: 480
    train_data_cat = ConcatDataset(
        [train_data_mask, train_data_no_mask, train_data_no_mask_hflip, train_data_no_mask_fflip,
         train_data_no_mask_rotate60_90,
         train_data_no_mask_rotate100_180, train_data_no_mask_rotate200_270, train_data_no_mask_gassion_blur,
         train_data_no_mask_invert])
    # 500: 240
    # train_data_cat = ConcatDataset(
    #     [train_data_mask, train_data_no_mask, train_data_no_mask_hflip, train_data_no_mask_fflip,
    #      train_data_no_mask_rotate60_90])

    #  500: 360
    # train_data_cat = ConcatDataset(
    #     [train_data_mask, train_data_no_mask, train_data_no_mask_hflip, train_data_no_mask_fflip,
    #      train_data_no_mask_rotate60_90,
    #      train_data_no_mask_rotate100_180, train_data_no_mask_rotate200_270])

    # train_dat without data-augmentaion
    #train_data_cat = ConcatDataset([train_data_mask, train_data_no_mask])
    train_loader = DataLoader(dataset=train_data_cat, batch_size=batch_size, shuffle=True)
    # train_data = torchvision.datasets.ImageFolder(root="./face/train", transform=data_transform["val"])
    # train_loader = DataLoader(dataset=train_data, sampler=ImbalancedDatasetSampler(train_data),
    #                           batch_size=batch_size, shuffle=False, num_workers=0)
    # load validation data
    validation_data = torchvision.datasets.ImageFolder(root="./face/validation", transform=data_transform["val"])
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, num_workers=0)
    # load test data
    test_data = torchvision.datasets.ImageFolder(root="./face/test", transform=data_transform["val"])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    train_size = len(train_data_cat)
    validation_size = len(validation_data)
    test_size = len(test_data)
    print(train_size)
    print(validation_size)
    print(test_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    alexnet = AlexNet()
    print(alexnet)
    alexnet.to(device)

    epoch = 60
    learning = 0.001
    optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning)
    loss = nn.CrossEntropyLoss()

    train_loss_all = []
    train_accur_all = []
    train_gmean_all = []
    validation_loss_all = []
    validation_accur_all = []
    validation_gmean_all = []
    for i in range(epoch):
        train_loss = 0
        train_num = 0.0
        train_accuracy = 0.0
        train_loader = tqdm(train_loader)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        ###################
        # start the train #
        ###################
        alexnet.train()
        for step, data in enumerate(train_loader):
            img, target = data
            optimizer.zero_grad()
            outputs = alexnet(img.to(device))

            loss1 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            loss1.backward()
            optimizer.step()
            train_loss += abs(loss1.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            train_accuracy = train_accuracy + accuracy
            train_num += img.size(0)
            outputs_list = outputs.data.tolist()  # transfer outputs to list
            # calculate the TP, FN, FP, TN
            for j in range(batch_size):  # get the classification result
                if j >= len(outputs_list):
                    break
                if outputs[j] == 0 and target.data.tolist()[j] == 0:  # 0 is "with mask", 1 is "without mask"
                    TP += 1
                elif outputs[j] == 1 and target.data.tolist()[j] == 0:
                    FN += 1
                elif outputs[j] == 0 and target.data.tolist()[j] == 1:
                    FP += 1
                else:
                    TN += 1
        print(TP, TN, FN, FP)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        # calculate G_mean
        G_mean = math.sqrt(TPR * TNR)
        print("epoch：{} ， train-Loss：{} , train-accuracy：{},G_mean:{}".format(i + 1, train_loss / train_num,
                                                                              train_accuracy / train_num, G_mean))
        train_loss_all.append(train_loss / train_num)
        train_accur_all.append(train_accuracy.double().item() / train_num)
        train_gmean_all.append(G_mean)

        validation_loss = 0
        validation_accuracy = 0.0
        validation_num = 0
        alexnet.eval()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        with torch.no_grad():
            validation_loader = tqdm(validation_loader)
            for data in validation_loader:
                img, target = data

                outputs = alexnet(img.to(device))

                loss2 = loss(outputs, target.to(device))
                outputs = torch.argmax(outputs, 1)
                validation_loss = validation_loss + abs(loss2.item()) * img.size(0)
                accuracy = torch.sum(outputs == target.to(device))
                validation_accuracy = validation_accuracy + accuracy
                validation_num += img.size(0)
                outputs_list = outputs.data.tolist()  # transfer outputs to list
                # calculate the TP, FN, FP, TN
                for j in range(batch_size):  # get the classification
                    if j >= len(outputs_list):
                        break
                    if outputs[j] == 0 and target.data.tolist()[j] == 0:  # 0 is "with mask", 1 is "without mask"
                        TP += 1
                    elif outputs[j] == 1 and target.data.tolist()[j] == 0:
                        FN += 1
                    elif outputs[j] == 0 and target.data.tolist()[j] == 1:
                        FP += 1
                    else:
                        TN += 1
        print(TP, TN, FN, FP)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        G_mean = math.sqrt(TPR * TNR)

        print("validation-Loss：{} , validation-accuracy：{}, G_mean:{}".format(validation_loss / validation_num,
                                                                              validation_accuracy / validation_num,
                                                                              G_mean))
        validation_loss_all.append(validation_loss / validation_num)
        validation_accur_all.append(validation_accuracy.double().item() / validation_num)
        validation_gmean_all.append(G_mean)
    torch.save(alexnet.state_dict(), "alexnet.pth")

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    test_accuracy = 0
    test_num = 0
    ###################
    # start the test #
    ###################
    alexnet.eval()
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for data in test_loader:
            img, target = data
            outputs = alexnet(img.to(device))
            outputs = torch.argmax(outputs, 1)
            # print(outputs)
            # print(target)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)
            outputs_list = outputs.data.tolist()  # transfer outputs to list
            # calculate the TP, FN, FP, TN
            for j in range(batch_size):  # get the classification
                if j >= len(outputs_list):
                    break
                if outputs[j] == 0 and target.data.tolist()[j] == 0:  # 0 is "with mask", 1 is "without mask"
                    TP += 1
                elif outputs[j] == 1 and target.data.tolist()[j] == 0:
                    FN += 1
                elif outputs[j] == 0 and target.data.tolist()[j] == 1:
                    FP += 1
                else:
                    TN += 1
    print(TP, TN, FN, FP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    G_mean = math.sqrt(TPR * TNR)
    print("test-accuracy：{}, G_mean:{}".format(test_accuracy / test_num, G_mean))

    # draw the curve
    # plt.figure(figsize=(12, ))
    # plt.subplot(1, 3, 1)
    # plt.plot(range(epoch), train_loss_all, "ro-", label="Train loss")
    # plt.plot(range(epoch), validation_loss_all, "bs-", label="validation loss")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("Loss")
    plt.subplot(1, 2, 1)
    plt.plot(range(epoch), train_accur_all, "ro-", label="Train accur")
    plt.plot(range(epoch), validation_accur_all, "bs-", label="validation accur")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch), train_gmean_all, "ro-", label="Train G_mean")
    plt.plot(range(epoch), validation_gmean_all, "bs-", label="validation G_mean")
    plt.xlabel("epoch")
    plt.ylabel("G_mean")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
