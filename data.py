from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

# rgb
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, phase, csv_file, split_rate, transform=tv.transforms.Compose([tv.transforms.ToTensor()])):
        #     csv_file(string): Path to the csv file with annotations.
        #     root_dir(string): Directory with all the images.
        #     transform(callable, optional): Optional transform to be applied on a sample.
        self.phase = phase  # validation or train
        # sep=";" separate the column
        cell_frame = pd.read_csv(csv_file, sep=';')
        img_name = cell_frame.iloc[:, 0]
        img_name = list(img_name)
        # print(type(img_name))
        # start from the third parameter, since the second is neither crack nor inactive
        self.label_all_set = np.array(cell_frame.iloc[:, 2:])
        # split data into train and test data set
        train_img_name, test_img_name, train_label, test_label = train_test_split(img_name, self.label_all_set, test_size=split_rate, random_state=1)

        if self.phase == 'train':
            self.img_name = train_img_name
            self.label = train_label
        if self.phase == 'val':
            self.img_name = test_img_name
            self.label = test_label

        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # temp = image_path.iloc[:, 0] 不用这么麻烦，就在read csv file的时候使用sep=";"就能分开
        # temp = temp.split(";")
        # self.img_list = temp[0]
        # start from the third parameter, since the second is neither crack nor inactive
        # self.label = np.array(temp[2:])
        # ./ 当前文件夹
        image = gray2rgb(imread('./' + self.img_name[idx]))
        image = self.transform(image)
        label = torch.from_numpy(self.label[idx])
        label = label.float()
        sample = (image, label)
        return sample

    def __len__(self):
        return len(self.img_name)

    def pos_weight(self):
        crack = self.label_all_set[:, 0]
        inactive = self.label_all_set[:, 1]
        crack_weight = np.sum(1 - crack) / np.sum(crack)
        inactive_weight = np.sum(1 - inactive) / np.sum(inactive)
        # return a tensor of weights
        pos_weight = torch.tensor([crack_weight, inactive_weight])
        return pos_weight


csv_file = './train.csv'
split_rate = 0.25


def get_train_dataset():
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(mean=train_mean, std=train_std)])
    train_data_set = ChallengeDataset('train', csv_file, split_rate, transform)
    return train_data_set


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(mean=train_mean, std=train_std)])
    validation_data_set = ChallengeDataset('val', csv_file, split_rate, transform)
    return validation_data_set

