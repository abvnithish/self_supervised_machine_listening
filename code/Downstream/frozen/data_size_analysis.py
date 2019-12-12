# -*- coding: utf-8 -*-

from main import *

# Training data size
sizes = [10, 50, 250, 500, 950]
model_data = pickle.load(
    open('/home/jk6373/self_supervised_machine_listening/code/downstream/Training_Data_Index.pkl', 'rb'))

# Model Variables
model_path, model_suffix, _ = model_path_list[int(sys.argv[1])]
print(model_suffix)

# Analysis
results_dict = {}
results_dict['val'], results_dict['test'] = {}, {}
for i in tqdm(range(len(sizes))):
    #     train_idx = idx_train
    #     if sizes[i] != 'Full':
    sample_key_train = model_data[i]
    train_idx = np.isin(sample_key, sample_key_train)

    Train_dataset = CQTLoader(root_dir, sample_key[train_idx], Y_mask[train_idx], Y_true[train_idx])
    Train_loader = DataLoader(dataset=Train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=my_collate)

    if r3.match(model_suffix) is None:
        model = AudioConvNet(fc=Identity())
    else:
        model = resnet18
        model.fc = Identity()

    # Execute & get results
    train_acc_list, train_loss_list, train_f1_list, best_train_class_wise_results, best_train_prt, \
    val_acc_list, val_loss_list, val_f1_list, best_val_class_wise_results, best_val_prt, best_model_state_dict = \
        run_model(model, model_path, model_suffix, Train_loader)

    results_dict['val'][sizes[i]] = {
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
    }
    model.load_state_dict(best_model_state_dict)
    results_dict['test'][sizes[i]] = test_model(Test_loader, model)

# Full Dataset results
file_path = '/home/jk6373/self_supervised_machine_listening/code/downstream/frozen/model/complete_dataset/'
file_name = 'downstream_frozen_' + model_suffix
results_dict['val']['Full'] = torch.load(file_path + file_name)

results_path = '/home/jk6373/self_supervised_machine_listening/results/'
model_test_results = pickle.load(open(results_path + 'complete_dataset_results.pkl', 'rb'))
results_dict['test']['Full'] = model_test_results[model_suffix]['test']

# Save results
pkl_file_path = '/home/jk6373/self_supervised_machine_listening/code/downstream/frozen/model/limited_dataset/'
pickle.dump(results_dict, open(pkl_file_path + model_suffix + '_trng_data_size_results.pkl', 'wb'))

print('Analysis Complete')