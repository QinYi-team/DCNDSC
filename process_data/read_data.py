import numpy as np
import pandas as pd
import h5py

#读取原始数据
tr_data = pd.read_csv(r"D:\zts\Dealing_data\process_data\AC_Test\修改\WTG\(gai)nlz_WTG_ACtest_data.csv")
tr_label = pd.read_csv(r"D:\zts\Dealing_data\process_data\AC_Test\修改\WTG\WJ_ACtest_label.csv")


#转换为二维矩阵读取
list_tr_data = tr_data.values
list_tr_label = tr_label.values


signal_tr = np.array(list_tr_data)
label_tr = np.array(list_tr_label)


signal_tr = signal_tr[:,:3072]
label_tr = label_tr[:,1]

print(signal_tr.shape, label_tr.shape)



f = h5py.File('nlz_WTG_ACtest.h5', 'w')
f.create_dataset('X_tr', data=signal_tr)
f.create_dataset('Y_tr', data=label_tr)
