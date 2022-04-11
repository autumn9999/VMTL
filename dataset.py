import torch
import numpy as np
import random
from PIL import Image
import torch.utils.data as data


def make_dataset(d_class, image_list):
    images_class_list = []
    labels_class_list = []

    if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
    else:
        for class_nums in range(d_class):
            images = [val.split()[0] for val in image_list if int(val.split()[1]) == class_nums]
            lables = [int(val.split()[1]) for val in image_list if int(val.split()[1]) == class_nums]
            images_class_list.append(images)
            labels_class_list.append(lables)
    return images_class_list, labels_class_list

def feature_loader(path, dataset):
    if dataset == "office-home":
        path = 'Dataset/office-home_feature/' + path.split('office-home/')[-1]

    path = path.split('.')[0] + '.npy'
    return path

def make_dataset_test(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

class ImageList(object):
    def __init__(self, dataset, d_class, image_list, number=1):

        self.dataset = dataset
        self.num_task = len(image_list)
        self.images_class_list = []
        self.labels_class_list = []
        for task_image_list in image_list:
            images_class_list, labels_class_list = make_dataset(d_class, task_image_list)
            self.images_class_list.append(images_class_list)
            self.labels_class_list.append(labels_class_list)

        self.number = number
        self.loader = feature_loader
        self.class_nums = d_class
        assert len(self.images_class_list[0]) == len(self.images_class_list[1]) == len(
            self.images_class_list[2]) == len(self.images_class_list[3]), 'wrong with dtaset initialization'

        imgs_list = []
        labels_list = []
        for task in range(self.num_task):
            imgs_class = []
            labels_class = []
            for index in range(self.class_nums):
                current_class = self.images_class_list[task][index]
                imgs = []
                labels = []
                for i in range(len(current_class)):
                    element = current_class[i]
                    path = self.loader(element, self.dataset)
                    feature = np.load(path)
                    label = self.labels_class_list[task][index][0]
                    imgs.append(feature)
                    labels.append(label)
                imgs = np.array(imgs)
                imgs_class.append(imgs)
                labels_class.append(labels)
            imgs_list.append(imgs_class)
            labels_list.append(labels_class)

        self.imgs = imgs_list
        self.labels = labels_list

        new_dataset_list = []
        for task in range(self.num_task):
            task_order = range(self.num_task)
            related_list = list(task_order)
            related_list.remove(task)
            new_task_list = []
            for index in range(self.class_nums):
                new_class_list = []
                current_feature = torch.Tensor(imgs_list[task][index]).cuda()
                for r in related_list:
                    related_feature = torch.Tensor(imgs_list[r][index]).cuda()
                    attention = torch.mm(current_feature, related_feature.transpose(0,1))
                    max, _ = attention.max(1)
                    min, _ = attention.min(1)
                    normalization = max-min
                    normalization = normalization.unsqueeze(1)
                    attention = attention/(normalization+1e-9)
                    attention = torch.softmax(attention, 1)
                    feature_f = torch.mm(attention, related_feature).cpu().numpy()
                    new_class_list.append(feature_f)
                new_class_list = np.array(new_class_list).transpose((1,0,2))
                new_task_list.append(new_class_list)
            new_dataset_list.append(new_task_list)
        self.prior_features = new_dataset_list

    def __getitem__(self, index):

        imgs_list = []
        related_imgs_list = []
        labels_list = []

        for task in range(self.num_task):
            current_feature = self.imgs[task][index]
            related_feature = self.prior_features[task][index]

            order = range(current_feature.shape[0])
            sample_order = list(order)

            while len(sample_order) < self.number:
                sample_order = sample_order + sample_order

            random.shuffle(sample_order)
            train_order = sample_order[:self.number]

            now_current_class = current_feature[train_order]
            now_related_class = related_feature[train_order]
            label = self.labels[task][index][0]

            imgs_list.append(now_current_class)
            related_imgs_list.append(now_related_class)
            labels_list.append(label)

        return imgs_list, labels_list, related_imgs_list

    def __len__(self):
        return self.class_nums


class ImageList_test(object):
    def __init__(self, dataset, image_list, labels=None):
        self.dataset = dataset
        self.imgs = make_dataset_test(image_list, labels)
        self.loader = feature_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        path_new = self.loader(path, self.dataset)
        feature = np.load(path_new)
        return feature, target

    def __len__(self):
        return len(self.imgs)



