
### Importing Libraries
import os
import time
import random
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import librosa
from librosa.display import specshow
import scipy, IPython.display as ipd
# from utils import *
### Setting seed for reproducibility
random_state = 7
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

debug_flag=False

def debug(str):
    if debug_flag:
        print(str)
        
class Master():
    '''
    Base class for Capstone
    '''
    def __init__(self, device, root_dir, path_file = '/checksums', sr = 22050, batch_size = 64, val_split = 0.2, transform_prob = 0.5, reduce_two_class=False, mode='jigsaw', model=None):
          
        self.device = device
        self.root_dir = root_dir
        self.path_file = path_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform_prob = transform_prob
        self.sr = sr
        self.reduce_two_class = reduce_two_class
        self.mode = mode
        print(f'MODE: {mode}')
        print('Getting Train & Validation Datasets')
        self.get_datasets()
        print('\t --Done')
        print('Creating Train & Validation Dataloaders')
        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=self.batch_size, shuffle=True) for x in ['train', 'valid']}
        print(f'Length of Train dataloader: {len(self.dataloaders["train"])} \t Validation dataloader: {len(self.dataloaders["valid"])}')
        print('\t --Done')
        self.model = model
        self.model = torch.nn.DataParallel(self.model)
        print('\t --Done')
        print('Init actions done')

    def get_filenames(self):
        '''
        Returns a list with paths to different files
        '''
        self.filenames = []
        self.exclude_files = ['098/098567.mp3_cqt.npy', '098/098565.mp3_cqt.npy', '098/098569.mp3_cqt.npy']
        print(f'Excluding these {len(self.exclude_files)} files - {self.exclude_files}')
        if os.path.exists(self.root_dir + self.path_file):
            with open(self.root_dir + self.path_file) as fp:
                lines = fp.readlines()
                for line in lines:
                    track_id, file_name = line.split()
                    if os.path.exists(self.root_dir + file_name + '_cqt.npy'):
                        if file_name + '_cqt.npy' not in self.exclude_files:
                            self.filenames.append(file_name + '_cqt.npy')
        print(f'There are a total of {len(self.filenames) + 1} music files in the root directory')
        return self.filenames    

    def get_datasets(self):
        '''
        Returns the Torch Tensor Dataset object with the tracks from the root_dir
        '''
        self.get_filenames()
        self.waveforms = []

        indices = np.arange(0, len(self.filenames), 1)
        random.seed(random_state)
        random.shuffle(indices)
        train_index  = int((1 - self.val_split) * len(self.filenames))

        self.train_files = np.array(self.filenames)[indices[:train_index]]
        self.val_files = np.array(self.filenames)[indices[train_index:]]

        train_dataset = fmaDataset(self.device, self.root_dir, self.train_files, sr = self.sr, transform_prob = self.transform_prob, reduce_two_class= self.reduce_two_class, mode=self.mode)
        val_dataset = fmaDataset(self.device, self.root_dir, self.val_files, sr = self.sr, transform_prob = self.transform_prob, reduce_two_class= self.reduce_two_class, mode=self.mode)

        self.datasets = {'train': train_dataset, 'valid': val_dataset}
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'valid']}

        print(f'# Training samples: {len(train_dataset)} \t # Validation samples: {len(val_dataset)}')


    def cqt_transform(self, waveform, display = True):
        '''
        Returns the CQT Spectogram and display first 10 seconds
        '''
        cqt = librosa.cqt(waveform)
        logscalogram = librosa.amplitude_to_db(np.abs(cqt))
        if display:
           specshow(logscalogram[:, :420])

        return cqt, logscalogram


    def train(self, num_epochs, learning_rate, print_every, verbose = True):
        '''
        Function to train the model
        '''

        print(f'Instantiating Optimzer, Loss Criterion, Scheduler')
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        print('\t --Done')

        print('Training started')
        self.model.train()

        start_time = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        acc_dict = {'train':[],'valid':[]}
        loss_dict = {'train':[],'valid':[]}

        for epoch in range(num_epochs):

            if verbose:
                if epoch % print_every == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}')
                    print('-' * 10)

            for phase in ['train','valid']:
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for iter_, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    if iter_ % 100:
                        print(f'Phase: {phase}   Iteration {iter_+1}/{len(self.dataloaders[phase])}', end="\r")

                    optimizer.zero_grad()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels.squeeze())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size()[0]
                    running_corrects += torch.sum(preds == labels.squeeze()).item()
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]

                if verbose:
                    if epoch % print_every == 0:
                        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # if save:
                #     if epoch % save_every == 0:
                #         if os.path.exists(os.path.join(root_dir,model_folder, 'modelStateDict.pt')):
                #             os.remove(os.path.join(root_dir,model_folder, 'modelStateDict.pt'))
                #         if os.path.exists(os.path.join(root_dir,model_folder, 'optimStateDict.pt')):
                #             os.remove(os.path.join(root_dir,model_folder, 'optimStateDict.pt'))
                #         torch.save(model.state_dict(), os.path.join(root_dir,model_folder, 'modelStateDict.pt'))
                #         torch.save(optimizer.state_dict(), os.path.join(root_dir,model_folder, 'optimStateDict.pt'))

                if phase == 'train':
                    loss_dict['train'].append(epoch_loss)
                    acc_dict['train'].append(epoch_acc)
                else:
                    loss_dict['valid'].append(epoch_loss)
                    acc_dict['valid'].append(epoch_acc)
                    scheduler.step(epoch_loss)

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            if epoch % print_every == 0:
                print('')

        time_elapsed = time.time() - start_time
        print(f'Training time: {int(time_elapsed / 60)}minutes {time_elapsed % 60}s')
        print(f'Best val Acc: {best_acc:4f}')


        fig = plt.figure()#figsize = (15, 12))
        plt.plot(loss_dict['train'])
        plt.plot(loss_dict['valid'])
        plt.title('Loss per epoch')
        train_patch = matplotlib.patches.Patch(color=sns.color_palette()[0], label= 'Train')
        valid_patch = matplotlib.patches.Patch(color=sns.color_palette()[1], label= 'Valid')
        plt.legend(handles=[train_patch, valid_patch])
        # plt.savefig(os.path.join(root_dir,model_folder, 'EpochWiseLoss_' + phase))
        plt.show()


        fig = plt.figure()
        plt.plot(acc_dict['train'])
        plt.plot(acc_dict['valid'])
        plt.title('Accuracy per epoch')
        plt.legend(handles=[train_patch, valid_patch])
        # plt.savefig(os.path.join(root_dir,model_folder, 'EpochWiseAccuracy_' + phase))
        plt.show()

        self.model.load_state_dict(best_model_wts)


    def evaluate_performance(self, compute_train = False, compute_val = True):
        '''
        Function to evaluate the performance of the model on train/validation dataset
        '''

        model.eval()

        if compute_train:
            pass

        if compute_val:
            pass

    def save_model(self, PATH_TO_SAVE):

        torch.save(self.model.state_dict(), PATH_TO_SAVE)

class fmaDataset(Dataset):

# def __init__(self, device, cqt_waveforms, transform_type, transform_prob):
    def __init__(self, device, root_dir, files, sr, transform_prob, reduce_two_class, mode):

        self.device = device
        self.root_dir = root_dir
        self.files = files
        self.sr = sr
        self.transform_prob = transform_prob
        self.reduce_two_class = reduce_two_class
        self.mode = mode
        self.permutation_mappings = [
            ([0,1,2],[1,0,0,0,0,0],0),
            ([0,2,1],[0,1,0,0,0,0],1),
            ([1,0,2],[0,0,1,0,0,0],2),
            ([1,2,0],[0,0,0,1,0,0],3),
            ([2,0,1],[0,0,0,0,1,0],4),
            ([2,1,0],[0,0,0,0,0,1],5)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #file = np.random.choice(files)
        logscalogram = np.load(self.root_dir + self.files[idx])
        try:
            assert logscalogram[:, :1280].shape == (84, 1280)
        except:
            print(self.files[idx])
            print(logscalogram[:, :1280].shape)
        
        logscalograms = [logscalogram[:, :420], logscalogram[:, 430: 850], logscalogram[:, 860:1280]]
        chosen_split = logscalograms[np.random.randint(0, 3)]
        if self.mode == 'jigsaw':
            jigsaw_splits = [np.split(chosen_split, 3, axis=1)][0] ## [0] is to flatten the array. [1] does not have any element
            if self.reduce_two_class:
                flip = np.random.rand()
                if flip < 0.5:
                    chosen_perm = self.permutation_mappings[0]
                    label = torch.tensor(0)
                else:
                    chosen_perm = self.permutation_mappings[np.random.randint(1, 6)]
                    label = torch.tensor(1)
            else:
                chosen_perm = self.permutation_mappings[np.random.randint(0, 6)]
                label = torch.tensor(chosen_perm[2])
            data = torch.FloatTensor(np.array([jigsaw_splits[x][:,:-5] for x in chosen_perm[0]])) ## Trimming last 5 pixels to avoid common borders
        #yield data.to(self.device), label.to(self.device)
        elif self.mode == 'flip':
            if np.random.rand() >= 0.5:
                data = np.flip(chosen_split, axis = 1).copy()
                label = torch.zeros(1).type(torch.LongTensor)
            else:
                data = chosen_split
                label = torch.ones(1).type(torch.LongTensor)
            data = torch.FloatTensor(data).unsqueeze(0)
        debug(data.shape)
        debug(label.shape)
        return data.to(self.device), label.to(self.device)








