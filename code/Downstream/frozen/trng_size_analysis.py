# -*- coding: utf-8 -*-

import os
import re
import sys
import pickle
import numpy as np
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import *

np.random.seed(7)
torch.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
import torchvision.models as models
resnet18 = models.resnet18().to(device)
alexnet = models.alexnet().to(device)
vgg16 = models.vgg16().to(device)
squeezenet = models.squeezenet1_0().to(device)
densenet = models.densenet161().to(device)
inception = models.inception_v3().to(device)
    
#####
DATA_ROOT = '/beegfs/bva212/openmic-2018'
OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'), allow_pickle=True)
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

len_data = Y_mask.shape[0]
idx_train = np.random.choice(len_data, int(len_data*0.7), replace=False)
remain_set = set(np.arange(len_data))-set(idx_train)
idx_test = np.random.choice(list(remain_set), int(len_data*0.1), replace=False)
idx_val = list(remain_set-set(idx_test))

Y_mask_train = Y_mask[idx_train]
Y_mask_val = Y_mask[idx_val]
Y_mask_test = Y_mask[idx_test]

label_train = Y_true[idx_train]
label_val = Y_true[idx_val]
label_test = Y_true[idx_test]

weights_train = np.sum(Y_mask_train, axis= 1)/20
new_weights_train = weights_train.reshape(-1,1)*Y_mask_train
weights_val = np.sum(Y_mask_val, axis= 1)/20
new_weights_val = weights_val.reshape(-1,1)*Y_mask_val
weights_test = np.sum(Y_mask_test, axis= 1)/20
new_weights_test = weights_test.reshape(-1,1)*Y_mask_test

class CQTLoader(Dataset):

    def __init__(self, root_dir, files, weights, label):
        self.weights = weights
        self.device = device
        self.root_dir = root_dir
        self.files = files
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        logscalogram = np.load(self.root_dir + self.files[idx]+'_cqt.npy')
        weight = self.weights[idx]
        label = self.label[idx]
        return {'logscalogram': logscalogram[np.newaxis, :], 'label': label[np.newaxis, :], 'weight': weight[np.newaxis,:]}

filenames = []
root_dir = '/beegfs/bva212/openmic-2018/cqt_full/'

BATCH_SIZE = 8

def my_collate(batch):
    data = np.concatenate([item['logscalogram'] for item in batch],axis=0)
    data = np.expand_dims(data, axis = 1)
    target = np.concatenate([item['label'] for item in batch],axis=0)
    weight = np.concatenate([item['weight'] for item in batch],axis=0)
    return [torch.from_numpy(data).float(), torch.from_numpy(target).float(), torch.from_numpy(weight).float()]

# Train_dataset = CQTLoader(root_dir, sample_key[idx_train], new_weights_train, label_train)
# Train_loader = DataLoader(dataset = Train_dataset, 
#                                               batch_size = BATCH_SIZE,
#                                               shuffle = True,
#                                           collate_fn = my_collate)

Val_dataset = CQTLoader(root_dir, sample_key[idx_val], new_weights_val, label_val)
Val_loader = DataLoader(dataset = Val_dataset, 
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                        collate_fn = my_collate)

Test_dataset = CQTLoader(root_dir, sample_key[idx_test], new_weights_test, label_test)
Test_loader = DataLoader(dataset = Test_dataset, 
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                        collate_fn = my_collate)

class AudioConvNet(nn.Module):

    def __init__(self, fc):
        '''
        Create the 5 Conv Layer Sound Net network architecture as per the paper - https://arxiv.org/pdf/1610.09001.pdf
        '''
        super(AudioConvNet, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels= 16, kernel_size = 5, stride = 2, padding = 5), 
                                nn.BatchNorm2d(num_features = 16), 
                                nn.ReLU(inplace = True),

                                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 5),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace = True),

                                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 5),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace = True),
                                nn.AvgPool2d(kernel_size = 3),

                                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, padding = 4),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace = True),

                                nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, padding = 4),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace = True),
                                nn.AvgPool2d(kernel_size = 3),

                                nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 3),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace = True),

                                nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 3),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(inplace = True),
                                nn.AdaptiveAvgPool2d(output_size = 1)
                                )
        
        self.fc = fc
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')


    def forward(self, input):
        conv_strips = []
        n_strips = input.shape[1]
        for strip in range(n_strips):
            conv_strip = input[:,strip]
            conv_strip = conv_strip.unsqueeze(1)
            conv_strips.append(self.conv_layers(conv_strip))

        concat_out=torch.cat(conv_strips,1)
        return self.fc(np.squeeze(concat_out))

# Function for testing the model
def test_model(loader, model):
    correct = 0
    total_loss = 0
    total = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        for spectrogram, target, weight in loader:
            spectrogram_batch, target_batch, weight_batch = spectrogram.to(device), target.to(device), weight.to(device)
            outputs = model(spectrogram_batch)
#             print(label_batch.shape)
            predicted = (torch.sigmoid(outputs.data)>0.5).float()
            loss = F.binary_cross_entropy_with_logits(outputs, target_batch,
                                                  weight = weight_batch,
                                                  reduction='sum')
            total_loss += loss.item()
            total += weight_batch.shape[0]

            correct += ((weight_batch != 0).float()*(predicted.eq(target_batch.view_as(predicted)).float())).sum().item()
            total_num += (weight_batch != 0).sum().item()
    return (100 * correct / total_num), (total_loss/total)

def train_model(train_loader, val_loader, model, optimizer, scheduler, num_epochs):
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    best_val_acc = 0
    model.eval()
    for epoch in range(num_epochs):
        for spectrogram, target, weight in train_loader:
            spectrogram_batch, target_batch, weight_batch = spectrogram.to(device), target.to(device), weight.to(device)
            optimizer.zero_grad()
            outputs = model(spectrogram_batch)
#             print(label_batch.shape)
            loss = F.binary_cross_entropy_with_logits(outputs, target_batch,
                                                  weight = weight_batch,
                                                  reduction='mean')
#             print(loss)
            loss.backward()
            optimizer.step()
        train_acc, train_loss = test_model(train_loader, model)
        val_acc, val_loss = test_model(val_loader, model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_dict = deepcopy(model.state_dict())
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        scheduler.step(val_acc)
#         print("Epoch:{}, Validation Accuracy:{:.2f}, Training Acc: {:.2f}, Val Loss: {:.5f}, Train Loss: {:.5f}".format(epoch+1, val_acc, train_acc, val_loss, train_loss))
    return train_acc_list, train_loss_list, val_acc_list, val_loss_list, best_model_state_dict
        
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class SimpleMLP_Model(nn.Module):
    """
    SimpleMLP classification model
    """
    def __init__(self):
        
        super(SimpleMLP_Model, self).__init__()
        #self.linear1 = nn.Linear(49152,512)
        #self.linear1 = nn.Linear(1024,512)
        #self.linear2 = nn.Linear(512,256)
        #self.linear3 = nn.Linear(256,10)
        self.linear = nn.Linear(1024,20)
    
    
    def forward(self,x):
        #x = x.view(x.size(0), -1)
        #x = nn.Linear(x.size(0),1024)(x)
        #x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        #x = self.linear3(x)  
        x = self.linear(x)
        return x
#####

sizes = [10, 50, 250, 500, 1000]
model_path_list = [('','random_init'),
                              ('/beegfs/sc6957/capstone/models/20191105/snet2_jigsaw_large_checkpoint_model_5.pth', 'jigsaw_linear_chkpnt_5'),
                              ('/beegfs/sc6957/capstone/models/20191106/snet2_jigsaw_large_best_model.pth', 'jigsaw_linear_best'),
                              ('/beegfs/bva212/capstone/new_model/checkpoint_model.pth', 'time_reversal_chkpnt_15'),
                              ('/beegfs/bva212/capstone/new_model/best_model.pth', 'time_reversal_best'),
                              ('/beegfs/sc6957/capstone/models/20191116/snet2_jigsaw_large_best_model.pth', 'jigsaw_3_1_linear_best'),
                              ('/beegfs/sc6957/capstone/models/20191116/snet3_jigsaw_large_best_model.pth', 'jigsaw_3_1_best')
                              ]

model_data = pickle.load(open('/home/jk6373/self_supervised_machine_listening/code/downstream/List_indices_data_size_exp.pkl', 'rb'))

r1 = re.compile('conv_layers')
r2 = re.compile('\w*chkpnt\w*')

model_state_dict = None
results_dict = {}

# for model_path, model_suffix in model_path_list:
model_path, model_suffix = model_path_list[int(sys.argv[1])]
print(model_suffix)
if model_suffix != 'random_init':
    if r2.match(model_suffix) is None:
        model_state_dict = torch.load(model_path)['modelStateDict']
    else:
        model_state_dict = torch.load(model_path)['bestModelStateDict']

for i in tqdm(range(len(sizes))):
    idx_train = model_data[i]
    Y_mask_train = Y_mask[idx_train]
    label_train = Y_true[idx_train]
    weights_train = np.sum(Y_mask_train, axis= 1)/20
    new_weights_train = weights_train.reshape(-1,1)*Y_mask_train
    Train_dataset = CQTLoader(root_dir, sample_key[idx_train], new_weights_train, label_train)
    Train_loader = torch.utils.data.DataLoader(dataset = Train_dataset, 
                                                                      batch_size = BATCH_SIZE,
                                                                      shuffle = True,
                                                                  collate_fn = my_collate)

    # Prepare/load model
    model = AudioConvNet(fc=Identity())

    if model_state_dict is not None: 
        for key in list(model_state_dict.keys()):
            cs = r1.search(key)
            if cs is None:
                del model_state_dict[key]
            elif cs.start() != 0:
                model_state_dict[key[cs.start():]] = model_state_dict[key]
                del model_state_dict[key]
#             print(model, model.state_dict().keys())
        model.load_state_dict(model_state_dict)

    for param in model.parameters():
            param.requires_grad = False

    model.fc = SimpleMLP_Model()
#     print(model, model.state_dict())
    model.to(device)

    # Train
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=0.01, weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True, \
                                                           threshold=0.03, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)
    train_acc_list, train_loss_list, val_acc_list, val_loss_list, best_model_state_dict = train_model(Train_loader, Val_loader, \
                                                                                                      model, optimizer, scheduler, num_epochs=30)


    results_dict[sizes[i]] = {
        'train_acc_list': train_acc_list,
        'train_loss_list': train_loss_list,
        'val_acc_list': val_acc_list,
        'val_loss_list': val_loss_list,
        'model_state_dict': best_model_state_dict
    }

file_path = '/home/jk6373/self_supervised_machine_listening/code/downstream/model/complete_dataset/'
file_name = 'downstream_frozen_' + model_suffix
results_dict[1500] = torch.load(file_path+file_name)

pkl_file_path = '/home/jk6373/self_supervised_machine_listening/code/downstream/model/limited_dataset/'
pickle.dump(results_dict, open(pkl_file_path+model_suffix+'_trng_data_size_results.pkl'.format(sizes[i]),'wb'))
print('complete')