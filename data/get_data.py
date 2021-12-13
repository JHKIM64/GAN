import pickle
import tensorflow as tf
import numpy as np
train_data = {}
test_data = {}

with open('/home/intern01/jhk/data.pickle','rb') as f :
    train_data = pickle.load(f)

train_data = train_data[:,:,30:,50:]
print(train_data.shape)
with open('/home/intern01/jhk/1820obsv.pickle','rb') as f :
    test_data = pickle.load(f)
    test_data = test_data.numpy()

train_mask = []
for i in range(0,14975) :
    mask = np.isnan(train_data[i,0,:,:])
    train_data[i,0,mask] = 0
    mask[mask == True] = 0
    mask[mask == False] = 1
    mask = np.array([np.array([mask])])
    if i==0 : train_mask = mask
    else : train_mask = np.concatenate([train_mask,mask],axis=0)


print(train_mask.shape)
train_data = np.concatenate([train_data,train_mask],axis=1)
print(np.isnan(train_data[0,0,:,:]))
train_data.transpose((0,2,3,1))
print(train_data.shape)
C_max = np.max(train_data[:, 0, :, :])
C_min = np.min(train_data[:, 0, :, :])
U_max = np.max(train_data[:, 1, :, :])
U_min = np.min(train_data[:, 1, :, :])
V_max = np.max(train_data[:, 2, :, :])
V_min = np.min(train_data[:, 2, :, :])
print(C_max, C_min)
print('**********')
print(U_max, U_min)
print('**********')
print(V_max, V_min)
print('*****new*****')

train_data[:, 0, :, :] = (train_data[:, 0, :, :] - C_min) / (C_max - C_min)
train_data[:, 1, :, :] = (train_data[:, 1, :, :] - U_min) / (U_max - U_min)
train_data[:, 2, :, :] = (train_data[:, 2, :, :] - V_min) / (V_max - V_min)

test_data[:, 0, :, :] = (test_data[:, 0, :, :] - C_min) / (C_max - C_min)
test_data[:, 1, :, :] = (test_data[:, 1, :, :] - U_min) / (U_max - U_min)
test_data[:, 2, :, :] = (test_data[:, 2, :, :] - V_min) / (V_max - V_min)

tcmax = np.nanmax(test_data[:, 0, :, :])
tcmin = np.nanmin(test_data[:, 0, :, :])
tumax = np.max(test_data[:, 1, :, :])
tumin = np.min(test_data[:, 1, :, :])
tvmax = np.max(test_data[:, 2, :, :])
tvmin = np.min(test_data[:, 2, :, :])
print(tcmax, tcmin)
print('**********')
print(tumax, tumin)
print('**********')
print(tvmax, tvmin)
print(np.count_nonzero(np.isnan(test_data[199,0,:,:])))