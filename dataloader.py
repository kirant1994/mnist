import numpy as np
import pandas as pd

def get_data(val_split=0.2):
    path = '/home/kiran/Documents/Workspace/datasets/MNIST'
    train = pd.read_csv('{0:s}/train.csv'.format(path), low_memory=False).values
    X_train, y_train = train[int(len(train) * val_split):, 1:], train[int(len(train) * val_split):, 0]
    X_test, y_test = train[:int(len(train) * val_split), 1:], train[:int(len(train) * val_split), 0]
    return X_train / 255, X_test / 255, y_train , y_test

#X_test, y_test = test[:, 1:], test[:, 0]
#test = pd.read_csv('{0:s}/test.csv'.format(path), low_memory=False).values
