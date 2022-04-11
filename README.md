# Variational Multi-Task Learning with Gumbel-Softmax Priors
Code for paper “Variational Multi-Task Learning with Gumbel-Softmax Priors” accepted by NeurIPS2021.

## Set Up
### Prerequisites
 - Python 3.6.9
 - Pytorch 1.1.0
 - GPU: an NVIDIA Tesla V100
 
### Getting Started
Inside this repository, we mainly conduct comprehensive experiments on Office-Home. Download the dataset from the following link, and place it in [`Dataset/`](./Dataset/) directory. To split documents are obtained by randomly selecting 5\%, 10\%, and 20\% of samples from each task as the training set and use the remaining samples as the test set. The split documents used for the office-home dataset comes from [MRN](https://github.com/thuml/MTlearn). There are three training list files called train_5.txt, train_10.txt and train_20.txt and three test list files called test_5.txt, test_10.txt, test_20.txt which are corresponding to the training and test files of 5\% data, 10\% data and 20\% data. More splits like(30\%, 40\%, ...) used for the ablation study in our paper will be open in the public repository.
- Office-home; [[link]](https://www.hemanthdv.org/officeHomeDataset.html)

To extract the input features based on VGG16 by using the following command:
```
python feature_vgg16.py #gpu_id #split
```

## Experiments

### Evaluation
To evaluate with the trained model  by:
```
python test.py #gpu_id #split #load_log_file/split vmtl
```
The pre-trained parameters of VMTL can be found in [`Trained_model/`](./Trained_model/)directory.

Due to the space limitation, the pre-trained parameters of VMTL-AC would be available in the final repository.

### Training
To train the proposed VMTL by running the command:
```
python train.py #gpu_id #split #log vmtl
```

To train VMTL with amortized classifiers by:
```
python train.py #gpu_id #split #log vmtl_ac
```

We will open the detailed split documents and pre-trained parameters of our models for the other datasets in the final public repository.

This code is build based on the following repositories:

[Learning Multiple Tasks with Multilinear Relationship Networks](https://github.com/thuml/MTlearn) and [Variational Autoencoder with Gumbel-Softmax Distribution](https://github.com/YongfeiYan/Gumbel_Softmax_VAE)
