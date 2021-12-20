import pickle
import numpy as np
import os
import util.plotcuv as plt_cuv
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def from_pickle(part) :
    train_data = {}
    for i in range(1,part+1) :
        with open('/home/intern01/jhk/pickles/data_part'+str(i)+'.pickle','rb') as f :
            train_part = pickle.load(f)

        if i==1 : train_data = train_part
        else : train_data = np.concatenate([train_data,train_part])

    return train_data

def check_data() :
    train_data = from_pickle(15)

    print(train_data.shape)
    plt_cuv.image('pm25',train_data[0,:,20:-10,20:-30],)

check_data()