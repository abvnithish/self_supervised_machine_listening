# -*- coding: utf-8 -*-

from main import *

# Prepare/load model
model_path, model_suffix = model_path_list[int(sys.argv[1])]
print(model_suffix)

# Execute & get results
train_acc_list, train_loss_list, train_f1_list, best_train_class_wise_results, best_train_prt, \
                val_acc_list, val_loss_list, val_f1_list, best_val_class_wise_results, best_val_prt, best_model_state_dict = \
    run_model(AudioConvNet(fc=Identity()), model_path, model_suffix, Train_loader)

file_path = '/home/jk6373/self_supervised_machine_listening/code/downstream/frozen/model/complete_dataset/'
file_name = 'downstream_frozen_' + model_suffix
torch.save({
    'train_acc_list': train_acc_list,
    'train_loss_list': train_loss_list,
    'train_f1_list': train_f1_list,
    'train_class_wise_results': best_train_class_wise_results,
    'train_prt': best_train_prt,
    'val_acc_list': val_acc_list,
    'val_loss_list': val_loss_list,
    'val_f1_list': train_f1_list,
    'val_class_wise_results': best_val_class_wise_results,
    'val_prt': best_val_prt,
    'model_state_dict': best_model_state_dict
    }, file_path+file_name)

print('Training Complete')