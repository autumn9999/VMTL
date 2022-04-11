import torch
import numpy as np
import os
import argparse

project_path = "./"

from dataset import ImageList, ImageList_test

def experiment(config, num_task):
    model = config["model"]
    dset_loaders = config["loaders"]
    test_interval = config["test_interval"]  # 100
    log_name = config["log_name"]
    best_acc = 0.0
    best_iter_num = 0
    best_list = []

    for iter_num in range(1, 2):
        checkpoint = torch.load(log_name + '/best_model.pth.tar')
        # amortized
        if config["model_type"] == "vmtl_ac":
            model.shared_encoder.load_state_dict(checkpoint["shared_encoder"])
            model.shared_generator.load_state_dict(checkpoint["shared_generator"])
            model.save_center_feature[0] = checkpoint["save_center_feature0"]
            model.save_center_feature[1] = checkpoint["save_center_feature1"]
            model.save_center_feature[2] = checkpoint["save_center_feature2"]
            model.save_center_feature[3] = checkpoint["save_center_feature3"]
            model.gumbel_list[0].load_state_dict(checkpoint["gumbel_list0"])
            model.gumbel_list[1].load_state_dict(checkpoint["gumbel_list1"])
            model.gumbel_list[2].load_state_dict(checkpoint["gumbel_list2"])
            model.gumbel_list[3].load_state_dict(checkpoint["gumbel_list3"])
            model.shared_generator_bias.load_state_dict(checkpoint["shared_generator_bias"])
        elif config["model_type"] == "vmtl":
            model.shared_encoder.load_state_dict(checkpoint["shared_encoder"])
            model.specific_w_list[0].load_state_dict(checkpoint["specific_w_list0"])
            model.specific_w_list[1].load_state_dict(checkpoint["specific_w_list1"])
            model.specific_w_list[2].load_state_dict(checkpoint["specific_w_list2"])
            model.specific_w_list[3].load_state_dict(checkpoint["specific_w_list3"])
            model.gumbel_list[0].load_state_dict(checkpoint["gumbel_list0"])
            model.gumbel_list[1].load_state_dict(checkpoint["gumbel_list1"])
            model.gumbel_list[2].load_state_dict(checkpoint["gumbel_list2"])
            model.gumbel_list[3].load_state_dict(checkpoint["gumbel_list3"])

        if iter_num % test_interval == 1:
            epoch_acc_list = test(dset_loaders["test"], model)

            for i in range(num_task):
                print('Iter {:05d} Acc on Task {:d}: {:.3f}'.format(iter_num, i, epoch_acc_list[i]))

            if np.mean(epoch_acc_list) > best_acc:
                best_list = epoch_acc_list
                best_acc = np.mean(epoch_acc_list)
                best_iter_num = iter_num

            print('Now- val Acc on Iter {:05d}: {:.3f}'.format(iter_num, np.mean(epoch_acc_list)))
            print('Best val Acc on Iter {:05d}: {:.3f}'.format(best_iter_num, best_acc))
            print('Iter {:05d}: {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(best_iter_num, best_list[0], best_list[1],
                                                                             best_list[2], best_list[3]))

def test(loaders, model):
    accuracy_list = []
    iter_val = [iter(loader) for loader in loaders]
    for i in range(len(iter_val)):
        iter_ = iter_val[i]
        start_test = True

        for j in range(len(loaders[i])):
            inputs, labels = iter_.next()
            inputs = inputs.cuda()
            labels = labels.cuda()

            predicts, labels = model.test_model(inputs, labels, i)
            if start_test:
                all_predict = predicts.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_predict = torch.cat((all_predict, predicts.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

        right_number = torch.sum(torch.squeeze(all_predict).float() == all_label)
        accuracy_list.append(right_number.item() / float(all_label.size()[0]))
    return accuracy_list

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('split', type=str, nargs='?', default='train_5', help="train_5, train_10, train_20")
    parser.add_argument('log_name', type=str, nargs='?', default='log', help="log name")
    parser.add_argument('model_type', type=str, nargs='?', default='vmtl_ac', help="vmtl_ac or vmtl")
    parser.add_argument('bs_number', type=int, nargs='?', default=4, help="...?")
    parser.add_argument('d_latent', type=int, nargs='?', default=512, help="...?")
    parser.add_argument('basenet', type=str, nargs='?', default='vgg16', help="or resnet18")
    parser.add_argument('dropout_index', type=float, nargs='?', default='0.7', help="??")
    parser.add_argument('temp', type=float, nargs='?', default='1', help="temperature")
    parser.add_argument('anneal', type=str2bool, nargs='?', default='True', help="or True")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}

    config["model_type"] = args.model_type
    if config["model_type"] == "vmtl_ac":
        import model_vmtl_ac as main_model
    elif config["model_type"] == "vmtl":
        import model_vmtl as main_model

    all_parameters = [args.temp, args.anneal, args.d_latent, args.bs_number, args.dropout_index]
    config["temp"] = args.temp
    config["anneal"] = args.anneal
    config["d_latent"] = args.d_latent
    config["bs_number"] = args.bs_number
    config["dropout_index"] = args.dropout_index

    config["gpus"] = range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    config["num_iter"] = 15000
    config["test_interval"] = 50
    config["lr"] = {"init_lr": 0.0001, "gamma": 0.5, "stepsize": 3000}

    # split
    config["split"] = args.split
    train_txt = config["split"] + '.txt'
    test_txt = 'test' + train_txt.split('train')[-1]
    split_name = config["split"].split("_")[-1]

    # save files
    config["log_name"] = args.log_name
    # os.system("mkdir -p " + config["log_name"])
    config["file_out"] = open(config["log_name"] + "/train_log.txt", "w")
    # os.system("mkdir -p " + config["log_name"] + '/files')
    # os.system('cp %s %s' % ('src/*.py', os.path.join(config["log_name"], 'files')))
    # print(str(config) + '\n')
    # config["file_out"].write(str(config) + '\n')
    # config["file_out"].flush()

    # dataset is "office-home"
    config["basenet"] = 'vgg16'
    task_name_list = ["Art", "Clipart", "Product", "Real_World"]
    train_file_list = [os.path.join(project_path,"Dataset", "split", "office-home", task_name_list[i], train_txt) for i in range(len(task_name_list))]
    test_file_list = [os.path.join(project_path,"Dataset", "split", "office-home", task_name_list[i], test_txt) for i in range(len(task_name_list))]
    d_class = 65
    batch_size = {"train": d_class, "test": 24}

    # dataset initialization
    dsets = {"train": [], "test": []}
    dsets["train"] = ImageList("office-home", d_class,[open(train_file_list[i]).readlines() for i in range(len(task_name_list))],number=config["bs_number"])
    dsets["test"] = [ImageList_test("office-home", open(test_file_list[i]).readlines()) for i in range(len(task_name_list))]

    # dataloader
    dset_loaders = {"train": [], "test": []}
    # dset_loaders["train"] = torch.utils.data.DataLoader(dsets["train"], batch_size=batch_size["train"], shuffle=True,num_workers=4)
    for test_dset in dsets["test"]:
        dset_loaders["test"].append(torch.utils.data.DataLoader(test_dset, batch_size=batch_size["test"], shuffle=False, num_workers=4))
    config["loaders"] = dset_loaders

    # model initialization
    config["model"] = main_model.MainModel("office-home", split_name, len(task_name_list), config["basenet"], d_class,config["file_out"], config["lr"], all_parameters)

    experiment(config, len(task_name_list))
    print("start training")
    # config["file_out"].close()