import pickle
import tensorflow as tf
import numpy as np
train_data = {}
test_data = {}
import os

## w e s n
## [100, 135, 25, 50]

print('test_data : ')
def get_obs_mask() :
    observation_mask = []
    with open('/home/intern01/jhk/1820obsv.pickle', 'rb') as f:
        test_data = pickle.load(f)

    for idx in range(0, 15) :
        if idx == 0 :
            observation_mask = np.array([np.isnan(test_data[idx*1000,0,:-3,:])])
        else :
            observation_mask = np.concatenate([observation_mask, np.array([np.isnan(test_data[idx*1000,0,:-3,:])])],axis=0)

    print(observation_mask.shape)
    return observation_mask

def get_test_data(C_max, C_min, W_max, W_min) :
    print('test_data : ')
    with open('/home/intern01/jhk/1820obsv.pickle','rb') as f :
        test_data = pickle.load(f)

    test_data[:, 0, :, :] = (test_data[:, 0, :, :] - C_min) / (C_max - C_min)
    test_data[:, 1:, :, :] = (test_data[:, 1:, :, :] - W_min) / (W_max - W_min)

    test_mask = []
    for i in range(0, 17536):
        mask = np.isnan(test_data[i, 0, :, :])
        test_data[i, 0, mask] = -1 # np.nan
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
    with open('/home/intern01/jhk/data_80_20_daily.pickle', 'rb') as f:
        train_data = pickle.load(f)

    train_data = train_data[:, :, 20:-13, 20:-30]
    C_max = np.max(train_data[:, 0, :, :])
    W_max = np.max(train_data[:, 1:, :, :])
    W_min = np.min(train_data[:, 1:, :, :])

    train_data[:, 0, :, :] = (train_data[:, 0, :, :]) / (C_max)
    train_data[:, 1:, :, :] = (train_data[:, 1:, :, :] - W_min) / (W_max - W_min)

    train_data_real = np.copy(train_data)
    train_mask = []
    observation_mask = get_obs_mask()
    for i in range(0, 14975):
        mask = np.zeros(train_data[i,0,:,:].shape, dtype=bool)
        mask[observation_mask[int(i/1000)]] = True
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

    train_data_input = train_data.transpose((0, 2, 3, 1))
    blank_tile_r = np.zeros((14975, 3, 48, 1))
    train_data_real = np.concatenate([train_data_real, blank_tile_r], axis=3)
    train_data_real = train_data_real.transpose((0, 2, 3, 1))
    with open('train_input.pickle', 'wb') as f:
        pickle.dump(train_data_input, f, pickle.HIGHEST_PROTOCOL)
    with open('train_real.pickle', 'wb') as f:
        pickle.dump(train_data_real, f, pickle.HIGHEST_PROTOCOL)

    return train_data_input, train_data_real , C_max, C_min, W_max, W_min

get_train_data()