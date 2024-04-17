import os
import time
import numpy as np
import pandas as pd
from hyper_param import Hyper_param
from process_data import raw_dataset
from model import raw_model
from torch.utils.data import DataLoader
import torch
from tool import accuracy

Folder_Path = r'D:\zts\DCNDSC_open\model_state_dict'
os.chdir(Folder_Path)
file_list = os.listdir()


config = Hyper_param()
test_dataset = raw_dataset(train=True)
test_loader = DataLoader(test_dataset,  batch_size=config.batch_size)
print('test_num = %d' % len(test_dataset))
model = raw_model(config.model_param)


ac = np.array([])


for i in range(0, len(file_list)):
    t0 = time.time()
    print('test_name:%s' % (file_list[i]))
    model.load_state_dict(torch.load(Folder_Path + '\\' + file_list[i]))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is available')
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    acc, cf_matrix = accuracy(model, test_loader, device, error_analysis=True)
    # confusion(model, test_loader, device)
    print(cf_matrix)

    a = np.array([acc])
    ac = np.append(ac, a)

mean_acc = np.mean(ac)
max_acc = np.max(ac)
std_acc = np.std(ac)

record_data1 = np.c_[mean_acc, max_acc, std_acc]
record_data = np.r_[record_data1]
data = pd.DataFrame(record_data)
out_path = r'D:\zts\DCNDSC_open\record\CWRU.csv'
data.to_csv(out_path, index=False, header=False)




