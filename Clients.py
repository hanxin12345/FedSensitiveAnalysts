import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model
# from tensorflow import plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Model_init


class client(object):
    def __init__(self, ID, localData, localLabel, model_init):
        self.ID = str(ID)
        self.dataset = localData
        self.label = localLabel
        self.local_model = model_init
        self.local_vars = model_init.trainable_variables


    def new_vars(self):
        self.local_model.fit(self.dataset, self.label)
        self.local_model.save('./cloth_model_'+self.ID+'.h5')

        return self.local_model.trainable_variables
