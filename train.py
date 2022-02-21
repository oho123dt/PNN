from read_csv import *
import numpy as np

from keras.layers import Input, Add, Dense, Flatten, Reshape
from keras.models import Model
from keras import backend as K

#import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#from sklearn.model_selection import cross_val_score
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import pickle,sys
from tqdm import tqdm as tqdm
import tensorflow as tf
# tf.set_random_seed(10)
import warnings
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=FutureWarning)
import h5py
np.random.seed(10)
import pandas as pd

import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from joblib import dump, load

F_train,S_train,V_train,D_train,W_train,H_train = read_data('train.csv')

X= np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
Y = np.concatenate((W_train.reshape(-1,1),H_train.reshape(-1,1)),1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

indices = tf.range(start=0, limit=tf.shape(X_std)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)
X_std = tf.gather(X_std, shuffled_indices)
Y = tf.gather(Y, shuffled_indices)
X_train_std = X_std[0:29]
Y_train = Y[0:29]
X_valid_std = X_std[30:]
Y_valid = Y[30:]

dump(scaler, 'fix_scaler.bin', compress=True)

def negloglik(y_true, y_pred):
    return -y_pred.log_prob(y_true) 

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution
act = 'relu'
model = keras.Sequential([
    keras.layers.Dense(units=10, activation=act, input_shape=(input_shape,)),
    keras.layers.Dense(units = 2)])
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[])
model_cb=ModelCheckpoint('./custom_model.h5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=300,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train_std,Y_train,epochs=1000,batch_size=4,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_valid_std, Y_valid))

df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./custom_model.csv',index=False)
