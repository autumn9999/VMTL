import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from vmtl_utils import kl_criterion_softplus
from vmtl_utils import local_reparameterize_softplus
from vmtl_utils import gumbel_softmax


class task_shared_network(nn.Module):
    def __init__(self, d_feature, d_latent, device, dropout_index=0.7):
        super(task_shared_network, self).__init__()
        self.d_feature = d_feature
        self.d_latent = d_latent
        self.device = device
        self.dropout_index = dropout_index
        self.rho = -3
        self.dropout = nn.Dropout(p=self.dropout_index)

        self.fc1 = nn.Linear(d_feature, d_latent).cuda()
        self.fc2 = nn.Linear(d_latent, d_latent).cuda()

        self.phi_mu = nn.Parameter(
            torch.empty((d_latent, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            requires_grad=True)
        self.phi_logvar = nn.Parameter(
            self.rho + torch.empty((d_latent, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            requires_grad=True)
        self.phi_bias_mu = nn.Parameter(
            torch.empty((1, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            requires_grad=True)
        self.phi_bias_logvar = nn.Parameter(
            self.rho + torch.empty((1, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            requires_grad=True)

    def forward(self, x, z_repeat, usefor):
        if usefor == "z":
            x = self.dropout(x)

        x = f.elu(self.fc1(x))
        x = f.elu(self.fc2(x))

        z_mu = torch.mm(x, self.phi_mu) + self.phi_bias_mu
        phi_sigma = f.softplus(self.phi_logvar, beta=1, threshold=20)
        phi_bias_sigma = f.softplus(self.phi_bias_logvar, beta=1, threshold=20)
        z_var = torch.mm(x.pow(2), phi_sigma.pow(2)) + phi_bias_sigma.pow(2)

        if self.training:
            z = local_reparameterize_softplus(z_mu, z_var, z_repeat)  # z_repeat * bs * d_latent
        else:
            z = z_mu.expand(z_repeat, z_mu.shape[0], z_mu.shape[1])

        if usefor == "w":
            return z, z_mu, z_var

        elif usefor == "z":
            z = z.contiguous().view(-1, self.d_latent)
            return z, z_mu, z_var

class task_specific_gumbel(nn.Module):
    def __init__(self, device, d_task):
        super(task_specific_gumbel, self).__init__()
        self.device = device
        self.gumbel = nn.Parameter(
            nn.init.constant_(torch.empty((1, d_task - 1), device=self.device, dtype=torch.float32), 0.0),
            requires_grad=True)
        self.gumbel_w = nn.Parameter(
            nn.init.constant_(torch.empty((1, d_task - 1), device=self.device, dtype=torch.float32), 0.0),
            requires_grad=True)

    def forward(self, temp, gumbel_type):
        if gumbel_type == "feature":
            logits = self.gumbel
        elif gumbel_type == "classifier":
            logits = self.gumbel_w
        current_prior_weights = gumbel_softmax(logits, temp, False)
        probability = torch.sigmoid(logits)
        return current_prior_weights.transpose(0, 1), probability.transpose(0, 1)

class MainModel(object):
    def __init__(self, dataset, split_name, task_num, network_name, class_num, file_out, optim_param, all_parameters):
        self.dataset = dataset
        self.split_name = split_name
        self.temp, self.anneal, self.d_latent, self.num, self.dropout_index= all_parameters
        self.classifier_bias = True

        self.temp_min = 0.5
        self.ANNEAL_RATE = 0.00003
        self.device = 'cuda'
        self.train_cross_loss = 0.0
        self.train_kl_loss = 0.0
        self.train_kl_w_loss = 0.0
        self.train_kl_z_loss = 0.0
        self.train_total_loss = 0.0
        self.beta = 0
        self.target_CE = 0.0

        self.file_out = file_out
        self.print_interval = 10
        self.eta = 1e-07

        self.optim_param = optim_param
        for val in optim_param:
            self.optim_param[val] = optim_param[val]

        self.task_num = task_num
        self.network_name = network_name
        self.d_class = class_num
        self.d_feature = 4096
        # network initialization************************************************************
        self.shared_encoder = task_shared_network(self.d_feature, self.d_latent, self.device, self.dropout_index)
        parameter_encoder = [{"params": self.shared_encoder.parameters(), "lr": 1}]
        self.shared_generator = task_shared_network(self.d_feature, self.d_latent, self.device, self.dropout_index)
        parameter_generator = [{"params": self.shared_generator.parameters(), "lr": 1}]
        parameter_amortized = parameter_encoder + parameter_generator

        if self.classifier_bias:
            self.shared_generator_bias = task_shared_network(self.d_feature, 1, self.device, self.dropout_index)
            parameter_generator_bias = [{"params": self.shared_generator_bias.parameters(), "lr": 1}]
            parameter_amortized += parameter_generator_bias

        parameters_all = []
        self.gumbel_list = []
        self.optimizer_list = []

        self.save_center_feature = []
        self.save_z_mu = []
        self.save_z_var = []
        self.save_w_mu = []
        self.save_w_var = []
        self.save_b_mu = []
        self.save_b_var = []

        for i in range(self.task_num):
            parameters_all.append(parameter_amortized)
            self.gumbel_list.append(task_specific_gumbel(self.device, self.task_num))
            parameters_all[i] = parameters_all[i] + [{"params": self.gumbel_list[i].parameters(), "lr": 1}]
            self.optimizer_list.append(optim.Adam(parameters_all[i], lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005))

            self.save_center_feature.append(torch.zeros(self.d_class, self.d_feature).cuda())
            self.save_z_mu.append(torch.zeros(self.d_class, self.d_latent).cuda())
            self.save_z_var.append(torch.ones(self.d_class, self.d_latent).cuda())
            self.save_w_mu.append(torch.zeros(self.d_class, self.d_latent).cuda())
            self.save_w_var.append(torch.ones(self.d_class, self.d_latent).cuda())
            self.save_b_mu.append(torch.zeros(self.d_class, 1).cuda())
            self.save_b_var.append(torch.ones(self.d_class, 1).cuda())

        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1
        self.counter = 1
        self.current_lr = 0.0
        self.z_repeat = 10

        self.z_mu_prior = nn.Parameter(
            nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.z_var_prior = nn.Parameter(
            nn.init.constant_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32), 1),
            requires_grad=False)
        self.w_mu_prior = nn.Parameter(
            nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.w_var_prior = nn.Parameter(
            nn.init.constant_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32), 1),
            requires_grad=False)
        self.b_mu_prior = nn.Parameter(
            nn.init.zeros_(torch.empty((self.d_class, 1), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.b_var_prior = nn.Parameter(
            nn.init.constant_(torch.empty((self.d_class, 1), device=self.device, dtype=torch.float32), 1),
            requires_grad=False)

    def optimize_model(self, input_list, label_list, number, related_inputs):
        # update learning rate for different networks
        if self.optimizer_list[0].param_groups[0]["lr"] >= 0.000002:
            self.current_lr = self.optim_param["init_lr"] * (self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
        for optimizer in self.optimizer_list:
            for component in optimizer.param_groups:
                component["lr"] = self.current_lr * 1.0

        self.shared_encoder.train()
        self.shared_generator.train()
        if self.classifier_bias:
            self.shared_generator_bias.train()
        self.gumbel_list[number].train()

        x = input_list.view(self.num, self.d_class, -1)
        center_x = x.mean(0)

        label_class = label_list.view(self.num, self.d_class)[0]
        _, indices = torch.sort(label_class)

        if self.counter > self.task_num:
            sum_number = self.counter // self.task_num
            sum_center = self.save_center_feature[number] * sum_number
            self.save_center_feature[number] = (sum_center + center_x[indices]) / (sum_number + 1)
        else:
            self.save_center_feature[number] = center_x[indices]  # bs * d_feature

        z, z_mu, z_var = self.shared_encoder(input_list, self.z_repeat, "z")
        w, w_mu, w_var = self.shared_generator(self.save_center_feature[number], z.shape[0], "w")
        self.save_w_mu[number] = w_mu
        self.save_w_var[number] = w_var

        if self.classifier_bias:
            b, b_mu, b_var = self.shared_generator_bias(self.save_center_feature[number], z.shape[0],"w")  # 65*1
            self.save_b_mu[number] = b_mu
            self.save_b_mu[number] = b_var
            output = torch.bmm(w, z.unsqueeze(2)).squeeze(2) + b.squeeze(2)
        else:
            output = torch.bmm(w, z.unsqueeze(2)).squeeze(2)

        re_label_list = label_list.expand(self.z_repeat, label_list.shape[0]).contiguous().view(-1)
        cls_loss = self.criterion(output, re_label_list)

        # kl_divergence
        q_z_mu = z_mu
        q_z_var = z_var

        q_w_mu = w_mu
        q_w_var = w_var

        if self.classifier_bias:
            q_b_mu = b_mu
            q_b_var = b_var

        if self.counter < 5:
            p_z_mu = self.z_mu_prior[label_list]
            p_z_var = self.z_var_prior[label_list]

            p_w_mu = self.w_mu_prior
            p_w_var = self.w_var_prior

            if self.classifier_bias:
                p_b_mu = self.b_mu_prior
                p_b_var = self.b_var_prior
                kl_b = torch.sum(kl_criterion_softplus(q_b_mu, q_b_var, p_b_mu, p_b_var))

            kl_w = torch.sum(kl_criterion_softplus(q_w_mu, q_w_var, p_w_mu, p_w_var))
            kl_z = torch.mean(kl_criterion_softplus(q_z_mu, q_z_var, p_z_mu, p_z_var))
        else:
            task_order = range(self.task_num)
            task_list = list(task_order)
            task_list.remove(number)

            current_prior_weights_feat, probability_feat = self.gumbel_list[number](self.temp, "feature")
            current_prior_weights_clas, probability_clas = self.gumbel_list[number](self.temp, "classifier")

            p_z_mu = 0.0
            p_z_var = 0.0
            p_w_mu = 0.0
            p_w_var = 0.0
            if self.classifier_bias:
                p_b_mu = 0.0
                p_b_var = 0.0

            for i in range(len(task_list)):
                p_number = task_list[i]
                current_coefficient_feat = current_prior_weights_feat[i]  # 1*1
                current_coefficient_clas = current_prior_weights_clas[i]  # 1*1

                _, p_z_mu_element, p_z_var_element  = self.shared_encoder(related_inputs[i], self.z_repeat,"z")
                p_z_mu_element_ = current_coefficient_feat * p_z_mu_element.detach()
                p_z_var_element_ = current_coefficient_feat.pow(2) * p_z_var_element.detach()
                p_z_mu += p_z_mu_element_
                p_z_var += p_z_var_element_

                p_w_mu_element = self.save_w_mu[p_number]
                p_w_var_element = self.save_w_var[p_number]
                p_w_mu_element_ = current_coefficient_clas * p_w_mu_element.detach()
                p_w_var_element_ = current_coefficient_clas.pow(2) * p_w_var_element.detach()
                p_w_mu += p_w_mu_element_
                p_w_var += p_w_var_element_

                if self.classifier_bias:
                    p_b_mu_element = self.save_b_mu[p_number]
                    p_b_var_element = self.save_b_var[p_number]
                    p_b_mu_element_ = current_coefficient_clas * p_b_mu_element.detach()
                    p_b_var_element_ = current_coefficient_clas.pow(2) * p_b_var_element.detach()
                    p_b_mu += p_b_mu_element_
                    p_b_var += p_b_var_element_

            if self.classifier_bias:
                kl_b = torch.sum(kl_criterion_softplus(q_b_mu, q_b_var, p_b_mu, p_b_var))
            kl_w = torch.sum(kl_criterion_softplus(q_w_mu, q_w_var, p_w_mu, p_w_var))
            kl_z = torch.mean(kl_criterion_softplus(q_z_mu, q_z_var, p_z_mu, p_z_var))

        if self.classifier_bias:
            kl_w = kl_w + kl_b
        kl_w = self.beta * kl_w
        kl_z = self.eta * kl_z
        kl_loss = kl_w + kl_z

        # overall lossfunction
        loss = cls_loss + kl_loss

        # updates
        self.optimizer_list[number].zero_grad()
        loss.backward()
        self.optimizer_list[number].step()

        # -----------------------------------------------------------------------------------------------
        self.counter += 1
        batchtask = self.task_num
        if self.counter % batchtask == 0:
            self.iter_num += 1
        if self.iter_num % 10 == 0:
            self.beta += 1e-06
        if self.anneal:
            if self.iter_num % 1000 == 0:
                self.temp = np.max([self.temp * np.exp(-self.ANNEAL_RATE * self.iter_num), self.temp_min])
        # print
        self.train_cross_loss += cls_loss.item()
        self.train_kl_loss += kl_loss.item()
        self.train_kl_w_loss += kl_w.item()
        self.train_kl_z_loss += kl_z.item()
        self.train_total_loss += loss.item()

        if self.counter == 8:
            print("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}".format(
                    int(self.counter / batchtask), self.current_lr,
                    self.train_cross_loss / float(self.counter),
                    self.train_kl_loss / float(self.counter),
                    self.train_kl_w_loss / float(self.counter),
                    self.train_kl_z_loss / float(self.counter),
                    self.train_total_loss / float(self.counter)))
            self.file_out.write("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(
                    int(self.counter / batchtask), self.current_lr,
                    self.train_cross_loss / float(self.counter),
                    self.train_kl_loss / float(self.counter),
                    self.train_kl_w_loss / float(self.counter),
                    self.train_kl_z_loss / float(self.counter),
                    self.train_total_loss / float(self.counter)))

        if self.counter % (self.print_interval * batchtask) == 0:
            print("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}".format(
                    int(self.counter / batchtask), self.current_lr,
                    self.train_cross_loss / float(self.print_interval * batchtask),
                    self.train_kl_loss / float(self.print_interval * batchtask),
                    self.train_kl_w_loss / float(self.print_interval * batchtask),
                    self.train_kl_z_loss / float(self.print_interval * batchtask),
                    self.train_total_loss / float(self.print_interval * batchtask)))
            self.file_out.write("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(
                    int(self.counter / batchtask), self.current_lr,
                    self.train_cross_loss / float(self.print_interval * batchtask),
                    self.train_kl_loss / float(self.print_interval * batchtask),
                    self.train_kl_w_loss / float(self.print_interval * batchtask),
                    self.train_kl_z_loss / float(self.print_interval * batchtask),
                    self.train_total_loss / float(self.print_interval * batchtask)))
            self.file_out.flush()
            self.target_CE = self.train_cross_loss / float(self.print_interval * batchtask)
            self.train_cross_loss = 0
            self.train_kl_loss = 0
            self.train_kl_w_loss = 0
            self.train_kl_z_loss = 0
            self.train_total_loss = 0
        # -----------------------------------------------------------------------------------------------
    def test_model(self, input_, label, i):
        self.shared_encoder.eval()
        self.shared_generator.eval()
        if self.classifier_bias:
            self.shared_generator_bias.eval()

        z, _, _ = self.shared_encoder(input_, self.z_repeat, "z")
        w, _, _ = self.shared_generator(self.save_center_feature[i], z.shape[0],"w")

        if self.classifier_bias:
            b, _, _ = self.shared_generator_bias(self.save_center_feature[i], z.shape[0],"w")
            output = torch.bmm(w, z.unsqueeze(2)).squeeze(2) + b.squeeze(2)
        else:
            output = torch.bmm(w, z.unsqueeze(2)).squeeze(2)

        _, output_predict = torch.max(output, 1)
        re_label = label.expand(self.z_repeat, label.shape[0]).contiguous().view(-1)
        return output_predict, re_label