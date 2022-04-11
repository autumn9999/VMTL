import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torchvision import models
import pdb

project_path = "./"


import caffe_transform as caffe_t
from feature_vgg16_dataset import ImageList_feature

class Vgg16NoFc(nn.Module):
    def __init__(self):
        super(Vgg16NoFc, self).__init__()
        model_vgg = models.vgg16(pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.extract_feature_layers = nn.Sequential(self.features, self.classifier)
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.in_features
network_dict = {"vgg16": Vgg16NoFc}

def experiment(config):
    dset_loaders = config["loaders"]
    model = network_dict["vgg16"]().cuda() # base model
    count1 = feature_extractor(dset_loaders["test"], model)
    count2 = feature_extractor(dset_loaders["train"], model)
    print(count1)
    print(count2)
    print(count1+count2)

def feature_extractor(loaders, model):
    model.eval()
    accuracy_list = []
    iter_val = [iter(loader) for loader in loaders]
    count=0
    for i in range(len(iter_val)):
        iter_ = iter_val[i]
        for j in range(len(loaders[i])):
            inputs, path= iter_.next()
            print (path)
            inputs= inputs.cuda()
            features = model(inputs)
            features = features.squeeze()
            features = features.detach().cpu().numpy()
            name = path[0].split('.')[0]
            new_path = name.split('/')[:-1]
            new_p = ''
            for item in new_path:
                if item == '':
                    continue
                if item == 'office-home':
                    item = 'office-home_feature'
                new_p = new_p + '/' + item
            new_name = new_p + '/' + name.split('/')[-1]

            if not os.path.exists(new_p):
                os.makedirs(new_p)
            if not os.path.exists(new_name):
                count+=1

            pdb.set_trace()
            np.save(new_name, features)
            print(new_name)
    return count

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='15', help="device id to run")
    parser.add_argument('split', type=str, nargs='?', default='train_5', help="train_5, train_10, train_20")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}

    #---------------------------------
    config["gpus"] = range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    config["split"] = args.split
    config["num_tasks"] = 4
    config["batch_size"] = {"train":65, "test":24}
    batch_size = config["batch_size"]
    config["dset_name"] = "office-home"

    data_transforms = {
        'train': caffe_t.transform_train(resize_size=256, crop_size=224),
        'val': caffe_t.transform_train(resize_size=256, crop_size=224),
    }
    data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)

    train_txt = config["split"] + '.txt'
    test_txt = 'test' + train_txt.split('train')[-1]

    task_name_list = ["Art",  "Clipart", "Product", "Real_World"]
    train_file_list = [os.path.join(project_path, "Dataset", "split", "office-home", task_name_list[i], train_txt) for i in range(config["num_tasks"])]
    test_file_list = [os.path.join(project_path, "Dataset", "split", "office-home", task_name_list[i], test_txt) for i in range(config["num_tasks"])]
    dset_classes = range(65)

    # dataset initialization
    dsets = {"train":[], "test":[]}
    dsets["train"] = [ImageList_feature(open(train_file_list[i]).readlines(),  config["dset_name"], transform=data_transforms["train"]) for i in range(len(task_name_list))]
    dsets["test"] = [ImageList_feature(open(test_file_list[i]).readlines(),  config["dset_name"], transform=data_transforms["val9"]) for i in range(len(task_name_list))]

    # dataloader
    dset_loaders = {"train":[], "test":[]}
    for train_dset in dsets["train"]:
        dset_loaders["train"].append(torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=False, num_workers=4))
    for test_dset in dsets["test"]:
        dset_loaders["test"].append(torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=4))
    config["loaders"] = dset_loaders

    count=1
    experiment(config)


