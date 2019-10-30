
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
from torch.utils.tensorboard import SummaryWriter
### Setting seed for reproducibility
random_state = 7
np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

debug_flag=True

def debug(str):
    if debug_flag:
        print(str)
        
class Master():
    '''
    Base class for Capstone
    '''
    def __init__(self, device, root_dir, path_file = '/checksums', sr = 22050, batch_size = 64, val_split = 0.2, transform_prob = 0.5, reduce_two_class=False, mode='jigsaw', model=None, num_seconds = 10):
          
        self.device = device
        self.root_dir = root_dir
        self.path_file = path_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform_prob = transform_prob
        self.sr = sr
        self.reduce_two_class = reduce_two_class
        self.mode = mode
        self.num_seconds = num_seconds
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

        train_dataset = fmaDataset(self.device, self.root_dir, self.train_files, sr = self.sr, transform_prob = self.transform_prob, reduce_two_class= self.reduce_two_class, mode=self.mode, num_seconds=self.num_seconds)
        val_dataset = fmaDataset(self.device, self.root_dir, self.val_files, sr = self.sr, transform_prob = self.transform_prob, reduce_two_class= self.reduce_two_class, mode=self.mode, num_seconds=self.num_seconds)
        self.input_size = train_dataset.input_size
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


    def train(self, num_epochs, learning_rate, print_every, verbose = True, save = False, model_save_path = '/beegfs/sc6957/capstone/', checkpoint_every = 5):
        '''
        Function to train the model
        '''

        print(f'Instantiating Optimzer, Loss Criterion, Scheduler')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        print('\t --Done')

        print('Training started')
        self.model.train()

        start_time = time.time()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0

        self.acc_dict = {'train':[],'valid':[]}
        self.loss_dict = {'train':[],'valid':[]}

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

                    self.optimizer.zero_grad()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(inputs)
                        _, preds = torch.max(logits, 1)
                        loss = self.criterion(logits, labels.squeeze())

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size()[0]
                    running_corrects += torch.sum(preds == labels.squeeze()).item()
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]

                if verbose:
                    if epoch % print_every == 0:
                        print()
                        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    self.loss_dict['train'].append(epoch_loss)
                    self.acc_dict['train'].append(epoch_acc)
                else:
                    self.loss_dict['valid'].append(epoch_loss)
                    self.acc_dict['valid'].append(epoch_acc)
                    scheduler.step(epoch_loss)

                if phase == 'valid' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    if save:
                        self.save_model(os.path.join(model_save_path + 'best_model.pth'))
            
            if epoch % print_every == 0:
                print('')

            if save:
                if epoch % checkpoint_every == 0 and phase == 'valid':
                    if os.path.exists(os.path.join(model_save_path + 'checkpoint_model.pth')):
                        os.remove(os.path.join(model_save_path + 'checkpoint_model.pth'))
                    self.checkpoint_model(os.path.join(model_save_path + 'checkpoint_model.pth'), epoch)
                    print(f'Successfully checkpointed model after {epoch} epochs')

        time_elapsed = time.time() - start_time

        print(f'Training time: {int(time_elapsed / 60)}minutes {time_elapsed % 60}s')
        print(f'Best val Acc: {self.best_acc:4f}')

        
        fig = plt.figure()#figsize = (15, 12))
        plt.plot(self.loss_dict['train'])
        plt.plot(self.loss_dict['valid'])
        plt.title('Loss per epoch')
        train_patch = matplotlib.patches.Patch(color=sns.color_palette()[0], label= 'Train')
        valid_patch = matplotlib.patches.Patch(color=sns.color_palette()[1], label= 'Valid')
        plt.legend(handles=[train_patch, valid_patch])
        if save:
            plt.savefig(os.path.join(model_save_path, 'EpochWiseLoss_' + phase + '.svg'))
        plt.show()


        log_dir = os.path.join(model_save_path + 'logs')
        writer = SummaryWriter(log_dir=log_dir)
        for phase in ['train','valid']:
            n = len(self.acc_dict[phase])
            for n_iter in range(n):
                writer.add_scalar(f'Loss/{phase}', self.loss_dict[phase][n_iter], n_iter)
                writer.add_scalar(f'Accuracy/{phase}',self.acc_dict[phase][n_iter], n_iter)
        writer.close()
              
        fig = plt.figure()
        plt.plot(self.acc_dict['train'])
        plt.plot(self.acc_dict['valid'])
        plt.title('Accuracy per epoch')
        plt.legend(handles=[train_patch, valid_patch])
        if save:
            plt.savefig(os.path.join(model_save_path, 'EpochWiseAccuracy_' + phase + '.svg'))
        plt.show()

    def get_predictions(self, phase = 'valid', save = False, preds_save_path = '/beegfs/sc6957/capstone/predictions.pkl'):
        
        self.predictions = []
        for file_num, file_name in enumerate(self.datasets[phase].files):

            if file_num % 100:
                print(f'Phase: {phase}   Iteration {file_num+1}/{len(self.datasets[phase].files)}', end="\r")

            start_width, inputs, labels = self.datasets[phase].getitem_for_eval(file_num)
            inputs = inputs.unsqueeze(0).to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                logits, outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            self.predictions.append([file_name, 
                                    start_width,
                                    labels.item(), 
                                    logits.cpu().numpy().squeeze(), 
                                    outputs.cpu().numpy().squeeze(), 
                                    preds.item()])

            if save:
                with open(preds_save_path, 'wb') as f:
                    pkl.dump(self.predictions, f)

                print(f'Predictions list pickled at {preds_save_path}')

            return self.predictions


    def evaluate_performance(self, compute_train = False, compute_val = True, verbose = True):
        '''
        Function to evaluate the performance of the model on train/validation dataset
        '''

        if compute_train:
            return self.get_performance_stats('train', verbose)

        if compute_val:
            return self.get_performance_stats('valid', verbose)

    def get_performance_stats(self, phase, verbose):

        if phase == 'train':
            self.model.train(True)
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for iter_, (inputs, labels) in enumerate(self.dataloaders[phase]):

            if iter_ % 100:
                print(f'Phase: {phase}   Iteration {iter_+1}/{len(self.dataloaders[phase])}', end="\r")

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                logits, outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.squeeze()).item()

        overall_acc = running_corrects / self.dataset_sizes[phase]

        if verbose:
            print()
            print(f'Performance computed on {phase} dataset over {self.dataset_sizes[phase]} observations \n\tAcc: {overall_acc:.4f}')

        return overall_acc
    
    def checkpoint_model(self, PATH_TO_SAVE, epoch):

        self.checkpoint_model_dict = {'epoch': epoch,
                                      'batch_size': self.batch_size,
                                      'modelStateDict': self.model.state_dict(),
                                      'optimStateDict': self.optimizer.state_dict(),
                                      'bestValAcc': self.best_acc,
                                      'bestModelStateDict': self.best_model_wts,
                                      'loss_dict': self.loss_dict,
                                      'acc_dict': self.acc_dict
                                       }

        torch.save(self.checkpoint_model_dict, PATH_TO_SAVE)

    def save_model(self, PATH_TO_SAVE):

        self.model_dict = {'bestValAcc': self.best_acc,
                           'loss_dict': self.loss_dict,
                           'acc_dict': self.acc_dict,
                           'modelStateDict': self.model.state_dict(),
                           'optimStateDict': self.optimizer.state_dict(),
                            }

        torch.save(self.model_dict, PATH_TO_SAVE)


class fmaDataset(Dataset):

# def __init__(self, device, cqt_waveforms, transform_type, transform_prob):
    def __init__(self, device, root_dir, files, sr, transform_prob, reduce_two_class, mode, num_seconds):

        self.device = device
        self.root_dir = root_dir
        self.files = files
        self.sr = sr
        self.transform_prob = transform_prob
        self.reduce_two_class = reduce_two_class
        self.mode = mode
        self.num_seconds = num_seconds
        self.permutation_mappings = [
            ([0,1,2],[1,0,0,0,0,0],0),
            ([0,2,1],[0,1,0,0,0,0],1),
            ([1,0,2],[0,0,1,0,0,0],2),
            ([1,2,0],[0,0,0,1,0,0],3),
            ([2,0,1],[0,0,0,0,1,0],4),
            ([2,1,0],[0,0,0,0,0,1],5)
        ]
        self.input_size = self.__getitem__(0)[0].shape

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
        
        second_width = int(1280/30)
        desired_width = self.num_seconds * second_width   
        start_width = random.randint(0, logscalogram.shape[1] - desired_width)
        chosen_split = logscalogram[:, start_width: start_width + desired_width]

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
        #debug(data.shape)
        #debug(label.shape)
        return data.to(self.device), label.to(self.device)

    def getitem_for_eval(self, idx):
        #file = np.random.choice(files)
        logscalogram = np.load(self.root_dir + self.files[idx])
        try:
            assert logscalogram[:, :1280].shape == (84, 1280)
        except:
            print(self.files[idx])
            print(logscalogram[:, :1280].shape)
        
        second_width = int(1280/30)
        desired_width = self.num_seconds * second_width   
        start_width = random.randint(0, logscalogram.shape[1] - desired_width)
        chosen_split = logscalogram[:, start_width: start_width + desired_width]

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
        #debug(data.shape)
        #debug(label.shape)
        return start_width,data.to(self.device), label.to(self.device)






