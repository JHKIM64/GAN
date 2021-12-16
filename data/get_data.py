import pickle
import tensorflow as tf
import numpy as np
train_data = {}
test_data = {}
import os

## w e s n
## [100, 135, 25, 50]

print('test_data : ')
with open('/home/intern01/jhk/1820obsv.pickle', 'rb') as f:
    test_data = pickle.load(f)
    test_data = test_data.numpy()

observation_mask1 = np.isnan(test_data[1000,0,:-3,:])
observation_mask2 = np.isnan(test_data[2000,0,:-3,:])
observation_mask3 = np.isnan(test_data[3000,0,:-3,:])
observation_mask4 = np.isnan(test_data[4000,0,:-3,:])
observation_mask5 = np.isnan(test_data[5000,0,:-3,:])
observation_mask6 = np.isnan(test_data[6000,0,:-3,:])
observation_mask7 = np.isnan(test_data[7000,0,:-3,:])
observation_mask8 = np.isnan(test_data[8000,0,:-3,:])
observation_mask9 = np.isnan(test_data[9000,0,:-3,:])
observation_mask10 = np.isnan(test_data[10000,0,:-3,:])

print(test_data.shape, observation_mask1.shape)

def get_test_data(C_max, C_min, U_max, U_min, V_max, V_min) :
    print('test_data : ')
    with open('/home/intern01/jhk/1820obsv.pickle','rb') as f :
        test_data = pickle.load(f)
        test_data = test_data.numpy()

    test_data[:, 0, :, :] = (test_data[:, 0, :, :] - C_min) / (C_max - C_min)
    test_data[:, 1, :, :] = (test_data[:, 1, :, :] - U_min) / (U_max - U_min)
    test_data[:, 2, :, :] = (test_data[:, 2, :, :] - V_min) / (V_max - V_min)

    test_mask = []
    for i in range(0, 17536):
        mask = np.isnan(test_data[i, 0, :, :])
        test_data[i, 0, mask] = -1 # np.nan 대체
        mask[mask == True] = 0
        mask[mask == False] = 1
        mask = np.array([np.array([mask])])
        if i == 0:
            test_mask = mask
        else:
            test_mask = np.concatenate([test_mask, mask], axis=0)

    test_data = np.concatenate([test_data, test_mask], axis=1)
    blank_tile = np.zeros((17536,4,51,1))
    test_data = np.concatenate([test_data, blank_tile], axis=3)
    test_data = test_data[:, :, :-3, :]
    with open('test_input.pickle', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    return test_data



def get_train_data() :
    print('train_data : ')
    with open('/home/intern01/jhk/data.pickle', 'rb') as f:
        train_data = pickle.load(f)

    train_data = train_data[:, :, 30:-3, 50:]
    train_data_real = np.copy(train_data)
    train_mask = []
    for i in range(0, 14975):
        mask = np.zeros(train_data[i,0,:,:].shape, dtype=bool)
        mask[observation_mask1] = True
        train_data[i, 0, mask] = -1 # np.nan 대체
        mask[mask == True] = 0
        mask[mask == False] = 1
        mask = np.array([np.array([mask])])
        if i == 0:
            train_mask = mask
        else:
            train_mask = np.concatenate([train_mask, mask], axis=0)

    train_data = np.concatenate([train_data, train_mask], axis=1)

    blank_tile = np.zeros((14975,4,48,1))
    train_data = np.concatenate([train_data, blank_tile], axis=3)

    C_max = np.max(train_data[:, 0, :, :])
    C_min = np.min(train_data[:, 0, :, :])
    U_max = np.max(train_data[:, 1, :, :])
    U_min = np.min(train_data[:, 1, :, :])
    V_max = np.max(train_data[:, 2, :, :])
    V_min = np.min(train_data[:, 2, :, :])

    train_data[:, 0, :, :] = (train_data[:, 0, :, :] - C_min) / (C_max - C_min)
    train_data[:, 1, :, :] = (train_data[:, 1, :, :] - U_min) / (U_max - U_min)
    train_data[:, 2, :, :] = (train_data[:, 2, :, :] - V_min) / (V_max - V_min)

    train_data_input = train_data.transpose((0, 2, 3, 1))
    blank_tile_r = np.zeros((14975, 3, 48, 1))
    train_data_real = np.concatenate([train_data_real, blank_tile_r], axis=3)
    train_data_real = train_data_real.transpose((0, 2, 3, 1))
    with open('train_input.pickle', 'wb') as f:
        pickle.dump(train_data_input, f, pickle.HIGHEST_PROTOCOL)
    with open('train_real.pickle', 'wb') as f:
        pickle.dump(train_data_real, f, pickle.HIGHEST_PROTOCOL)
    return train_data_input, train_data_real , C_max, C_min, U_max, U_min, V_max, V_min