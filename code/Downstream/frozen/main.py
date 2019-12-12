# -*- coding: utf-8 -*-

# Import required packages
import os
import re
import sys
import json
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_recall_curve

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import *

# Set seed to get reproducible results
np.random.seed(7)
torch.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

###
import torchvision.models as models
resnet18 = models.resnet18().to(device)
alexnet = models.alexnet().to(device)
vgg16 = models.vgg16().to(device)
squeezenet = models.squeezenet1_0().to(device)
densenet = models.densenet161().to(device)
inception = models.inception_v3().to(device)
###

##### Data Loading
DATA_ROOT = '/beegfs/bva212/openmic-2018'
OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'), allow_pickle=True)
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
    class_map = json.load(f)

# len_data = Y_mask.shape[0]
# idx_train = np.random.choice(len_data, int(len_data*0.7), replace=False)
# remain_set = set(np.arange(len_data)) - set(idx_train)
# idx_test = np.random.choice(list(remain_set), int(len_data*0.1), replace=False)
# idx_val = list(remain_set - set(idx_test))

train_samples = pd.read_csv(os.path.join(DATA_ROOT,'/beegfs/bva212/openmic-2018/partitions/split01_train.csv'), names =['id']).to_numpy().squeeze()
test_samples = pd.read_csv(os.path.join(DATA_ROOT,'/beegfs/bva212/openmic-2018/partitions/split01_test.csv'), names =['id']).to_numpy().squeeze()

len_data = len(train_samples)
train_idx = np.random.choice(len_data, int(len_data*0.8), replace=False)
remain_set = list(set(np.arange(len_data))-set(train_idx))

idx_val = np.isin(sample_key, train_samples[remain_set])
idx_train = np.isin(sample_key, train_samples[train_idx])
idx_test = np.isin(sample_key, test_samples)

class CQTLoader(Dataset):

    def __init__(self, root_dir, files, mask, label):
        self.root_dir = root_dir
        self.files = files
        self.mask = mask
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        logscalogram = (np.load(self.root_dir + self.files[idx]+'_cqt.npy') - -24.3633)/14.2659
        logscalogram = np.array(np.split(logscalogram[:,:logscalogram.shape[1]-logscalogram.shape[1]%3], 3, axis=1))
        label = (self.label[idx] > 0.5).astype(int)
        mask = self.mask[idx]
        return {'logscalogram': logscalogram[np.newaxis, :], 'label': label[np.newaxis, :], 'mask': mask[np.newaxis, :]}


def my_collate(batch):
    data = np.concatenate([item['logscalogram'] for item in batch], axis=0)
    # data = np.expand_dims(data, axis = 1)
    
    target = np.concatenate([item['label'] for item in batch],axis=0)
    
    mask_sum = np.sum([item['mask'] for item in batch], axis=0)
    mask_sum = np.where(mask_sum == 0, 1, mask_sum)
    weight = np.concatenate([item['mask'] / mask_sum for item in batch], axis=0)
    
    return [torch.from_numpy(data).float(), torch.from_numpy(target).float(), torch.from_numpy(weight).float()]


root_dir = '/beegfs/bva212/openmic-2018/cqt_full/'
BATCH_SIZE = 32

Train_dataset = CQTLoader(root_dir, sample_key[idx_train], Y_mask[idx_train], Y_true[idx_train])
Train_loader = DataLoader(dataset = Train_dataset, 
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                            collate_fn = my_collate
                          )

Val_dataset = CQTLoader(root_dir, sample_key[idx_val], Y_mask[idx_val], Y_true[idx_val])
Val_loader = DataLoader(dataset = Val_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                        collate_fn = my_collate
                        )

Test_dataset = CQTLoader(root_dir, sample_key[idx_test], Y_mask[idx_test], Y_true[idx_test])
Test_loader = DataLoader(dataset = Test_dataset, 
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                        collate_fn = my_collate
                        )


# Classification Model
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
        self.concat_mlp_layer = nn.Sequential(nn.Linear(3072, 2048),
                                              nn.BatchNorm1d(num_features=2048),
                                              nn.ReLU(inplace=True),

                                              nn.Linear(2048, 1024),
                                              nn.BatchNorm1d(num_features=1024),
                                              nn.ReLU(inplace=True),

                                              nn.Linear(1024, 256),
                                              nn.BatchNorm1d(num_features=256),
                                              nn.ReLU(inplace=True),
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
        # output = self.fc(np.squeeze(concat_out))

        # ConvNet-Ext
        out = self.concat_mlp_layer(concat_out.view(concat_out.shape[0], -1))
        output = self.fc(out.view(out.shape[0], -1))

        return output

# Function for calculating results metrics
def test_model(loader, model):
    
    # Declaration
    correct = 0
    total_loss = 0
    total = 0
    total_num = 0
    actual_arr = []
    predicted_arr = []
    weight_array = []
    
    model.eval()
    with torch.no_grad():
        wgt, tgt, preds, pred_probs = torch.zeros((1, len(class_map))).to(device), torch.zeros((1, len(class_map))).to(device), \
                                            torch.zeros((1, len(class_map))).to(device), torch.zeros((1, len(class_map))).to(device)
        for spectrogram, target, weight in loader:
            spectrogram_batch, target_batch, weight_batch = spectrogram.to(device), target.to(device), weight.to(device)
            outputs = model(spectrogram_batch)
            predicted = (torch.sigmoid(outputs.data)>0.5).float()
            loss = F.binary_cross_entropy_with_logits(outputs, target_batch,
                                                  weight = weight_batch,
                                                  reduction='sum')

            actual_arr.extend(target.view(1, -1).squeeze().numpy().astype(int).tolist())
            predicted_arr.extend(predicted.view(1, -1).squeeze().cpu().numpy().astype(int).tolist())

            total_loss += loss.item()
            total += weight_batch.shape[0]

            correct += ((weight_batch != 0).float()*(predicted.eq(target_batch.view_as(predicted)).float())).sum().item()
            total_num += (weight_batch != 0).sum().item()
            weight_array = np.concatenate((weight_array,(weight != 0).reshape(-1).numpy().astype(int)))
            
            wgt = torch.cat((wgt, weight_batch), dim=0)
            tgt = torch.cat((tgt, target_batch), dim=0)
            preds = torch.cat((preds, predicted), dim=0)
            pred_probs = torch.cat((pred_probs, torch.sigmoid(outputs.data)), dim=0)

        # Results
        o_acc = (100 * correct / total_num)
        o_loss = (total_loss/total)
        o_f1 = f1_score(actual_arr, predicted_arr, average='micro', sample_weight=weight_array)
        class_wise_results, o_prt = {}, {}
        for instrument, label in class_map.items():
            y_true = tgt[(wgt[:,label] != 0),label].cpu()
            y_preds = preds[(wgt[:,label] != 0),label].cpu()

            class_wise_results[instrument] = classification_report(y_true, y_preds, output_dict = True)

            o_prt[instrument] = precision_recall_curve(tgt[(wgt[:, label] != 0), label].cpu(), \
                                                               pred_probs[(wgt[:, label] != 0), label].cpu())

    return o_acc, o_loss, o_f1, class_wise_results, o_prt


def train_model(train_loader, val_loader, model, optimizer, scheduler, num_epochs):
    
    # Declaration
    train_acc_list = []
    train_loss_list = []
    train_f1_list = []
    val_acc_list = []
    val_loss_list = []
    val_f1_list = []
    best_val_acc = 0
    
    model.eval()
    for epoch in range(num_epochs):
        for spectrogram, target, weight in train_loader:
            spectrogram_batch, target_batch, weight_batch = spectrogram.to(device), target.to(device), weight.to(device)
            optimizer.zero_grad()
            outputs = model(spectrogram_batch)
            loss = F.binary_cross_entropy_with_logits(outputs, target_batch,
                                                  weight = weight_batch,
                                                  reduction='sum')
            loss.backward()
            optimizer.step()
        
        # Get results
        train_results = test_model(train_loader, model)
        val_results = test_model(val_loader, model)
        train_acc_list.append(train_results[0])
        train_loss_list.append(train_results[1])
        train_f1_list.append(train_results[2])
        val_acc_list.append(val_results[0])
        val_loss_list.append(val_results[1])
        val_f1_list.append(val_results[2])
        
        # Store best model
        if val_results[0] > best_val_acc:
            best_val_acc = val_results[0]
            best_model_state_dict = deepcopy(model.state_dict())
            best_train_class_wise_results = deepcopy(train_results[3])
            best_train_prt = deepcopy(train_results[4])
            best_val_class_wise_results = deepcopy(val_results[3])
            best_val_prt = deepcopy(val_results[4])
            
        scheduler.step(val_results[0])

        print("Epoch:{}".format(epoch + 1))
        print("Training - Accuracy: {:.2f}, Loss: {:.5f}, F1: {:.2f}".format(train_results[0], train_results[1],
                                                                                    train_results[2]))
        print("Validation - Accuracy: {:.2f}, Loss: {:.5f}, F1: {:.2f}".format(val_results[0], val_results[1],
                                                                                          val_results[2]))
        
    return train_acc_list, train_loss_list, train_f1_list, best_train_class_wise_results, best_train_prt, \
                val_acc_list, val_loss_list, val_f1_list, best_val_class_wise_results, best_val_prt, best_model_state_dict
        
        
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class SimpleMLP_Model(nn.Module):
    """
    SimpleMLP classification model
    """
    def __init__(self, in_ftr=1024, out_ftr=20):
        
        super(SimpleMLP_Model, self).__init__()
        #self.linear1 = nn.Linear(49152,512)
        #self.linear1 = nn.Linear(1024,512)
        #self.linear2 = nn.Linear(512,256)
        #self.linear3 = nn.Linear(256,10)
        self.linear = nn.Linear(in_ftr, out_ftr)
    
    
    def forward(self,x):
        #x = x.view(x.size(0), -1)
        #x = nn.Linear(x.size(0),1024)(x)
        #x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        #x = self.linear3(x)  
        x = self.linear(x)
        return x
#####

# Hyperparameters
lr = 0.01
num_epochs = 45

# Model Variables
model_path_list = [('','random_init', 'Random ConvNet'),
                    ('/beegfs/bva212/capstone/new_model/best_model.pth', 'time_reversal_best', 'Time Reversal ConvNet'),
                    ('/beegfs/sc6957/capstone/models/20191116/snet3_jigsaw_large_best_model.pth', 'jigsaw_3_1_best', 'Jigsaw 3*1s ConvNet'),
                    # ('/beegfs/sc6957/capstone/models/20191105/snet2_jigsaw_large_checkpoint_model_5.pth', 'jigsaw_10_3_wfc_chkpnt_5'),
                    ('/beegfs/sc6957/capstone/models/20191106/snet2_jigsaw_large_best_model.pth', 'jigsaw_10_3_wfc_best', 'Jigsaw 3*3s Convnet'),
                    # ('/beegfs/sc6957/capstone/models/20191116/snet2_jigsaw_large_best_model.pth', 'jigsaw_3_1_wfc_best'),
                    # ('/beegfs/bva212/capstone/new_model/checkpoint_model.pth', 'time_reversal_chkpnt_15'),
                    ('', 'random_ext_init', 'Random ConvNet-ext'),
                    ('/beegfs/sc6957/capstone/models/20191116/snet2_jigsaw_large_best_model.pth', 'jigsaw_3_1_ext_best', 'Jigsaw 3*1s ConvNet-ext'),
                    ('/beegfs/sc6957/capstone/models/20191106/snet2_jigsaw_large_best_model.pth', 'jigsaw_10_3_ext_best', 'Jigsaw 3*3s ConvNet-ext'),
                    ('', 'random_resnet_init', 'Random ResNet18'),
                    ('/beegfs/sc6957/capstone/models/20191123/resnet_jigsaw_large_best_model.pth', 'jigsaw_3_1_resnet_best', 'Jigsaw 3*1s ResNet18'),
                    ('/beegfs/sc6957/capstone/models/20191123/resnet_jigsaw_10_large_best_model.pth', 'jigsaw_10_3_resnet_best', 'Jigsaw 3*3s ResNet18')
                    ]

r1 = re.compile('conv_layers')
r2 = re.compile('\w*chkpnt\w*')
r3 = re.compile('\w*resnet\w*')
r4 = re.compile('mlp_layer\w*')
r5 = re.compile('fc')
r6 = re.compile('random\w*')


def run_model(model, model_path, model_suffix, train_loader):
    
    model_state_dict = None
    if r6.match(model_suffix) is None:
        if r2.match(model_suffix) is None:
            model_state_dict = torch.load(model_path)['modelStateDict']
        else:
            model_state_dict = torch.load(model_path)['bestModelStateDict']

    if model_state_dict is not None: 
        for key in list(model_state_dict.keys()):
            # ConvNet
            # cs = r1.search(key)
            # if cs is None:
            #     del model_state_dict[key]
            # elif cs.start() != 0:
            #     model_state_dict[key[cs.start():]] = model_state_dict[key]
            #     del model_state_dict[key]

            if r3.match(model_suffix) is None:
                # ConvNet-Ext
                n_ftr = 256
                if r4.match(key) is not None:
                    del model_state_dict[key]
            else:
                # ResNet
                n_ftr = 512
                cs = r5.search(key)
                if cs is None:
                    model_state_dict[key[len('resnet.'):]] = model_state_dict[key]
                    del model_state_dict[key]
                elif cs.start() != 0:
                    del model_state_dict[key]
        model.load_state_dict(model_state_dict)

    # Freeze
    for param in model.parameters():
            param.requires_grad = False

    # ConvNet: 1024, ConvNet-Ext: 256, ResNet: 512
    if r3.match(model_suffix) is None:
        # ConvNet-Ext
        n_ftr = 256
    else:
        # ResNet
        n_ftr = 512
    model.fc = SimpleMLP_Model(in_ftr=n_ftr, out_ftr=20)
    model.to(device)

    # Train
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=lr, weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True, \
                                                           threshold=0.03, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)
    
    # Output results
    return train_model(train_loader, Val_loader, model, optimizer, scheduler, num_epochs)