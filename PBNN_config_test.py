from read_csv import *

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm as tqdm

import h5py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras.backend import mean, std, max, get_value

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load

######### IMPORT DATA ###############

# F_train,S_train,V_train,D_train,W_train,H_train = read_data('train.csv')

# X= np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
# Y = np.concatenate((W_train.reshape(-1,1),H_train.reshape(-1,1)),1)
# F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
# X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
# Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

# scaler = MinMaxScaler(feature_range = (-1,1))
# X_std = scaler.fit_transform(X)
# X_test_norm = scaler.transform(X_test)

# X_train_std = X_std[0:25]
# Y_train = Y[0:25]
# X_valid_std = X_std[26:]
# Y_valid = Y[26:]

######### IMPORT DATA ###############
F_train,S_train,V_train,D_train,W_train,H_train = read_data('train.csv')

X= np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
Y = np.concatenate((W_train.reshape(-1,1),H_train.reshape(-1,1)),1)
F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

scaler = MinMaxScaler(feature_range = (-1,1))
X_std = scaler.fit_transform(X)
X_test_norm = scaler.transform(X_test)

X_train_std = X_std[0:25]
Y_train = Y[0:25]
X_valid_std = X_std[26:]
Y_valid = Y[26:]

dump(scaler, 'std_scaler.bin', compress=True)

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution
act = 'relu'

######### CALL BACK AND LOSS FUNCTION ##########

early_cb=EarlyStopping(monitor='val_loss', patience=500,verbose=0, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=0, patience=20, min_lr=0.001)
cb = [early_cb, reduce_lr]

def NLL(y_true, y_pred):
    return -y_pred.log_prob(y_true) 

######### MODEL DEFINITION ##############
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(   
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = Sequential(
        [
            tfpl.VariableLayer(
                tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfpl.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

best_w_err = tf.zeros(Y_test.shape[0])
best_h_err = tf.zeros(Y_test.shape[0])
w_err = tf.zeros(Y_test.shape[0])
h_err = tf.zeros(Y_test.shape[0])
best_w_Eavg = 10
best_h_Eavg = 10
best_layer = 9
best_dense = 3

train_time = 30

# train model several time and saving the best weight with lowest %ER

for i in range (9,10):
    for k in range (6,7):
        for j in range (train_time):
            print('Hidden size: '+str(i) + '-' + str(k)+' Train time no: '+str(j))
            model = Sequential ([
                tfpl.DenseVariational (
                    units = i,  input_shape = (input_shape,),
                    make_prior_fn = prior, make_posterior_fn = posterior,
                    kl_weight = 1 / X_train_std.shape[0], activation = 'sigmoid'
                ),  
                tfk.layers.Dense (
                    units = k, activation = 'relu'
                ),              
                tfk.layers.Dense (
                    units = tfpl.MultivariateNormalTriL.params_size(2),
                ),
                tfpl.MultivariateNormalTriL(2)
            ])
            model.compile(loss=NLL, optimizer= RMSprop(learning_rate = 0.01))              # RMSprop(learning_rate = 0.005)

            history = model.fit(X_train_std,Y_train,epochs=2000, batch_size=5, verbose=0, callbacks= cb,shuffle=True,validation_data=(X_valid_std, Y_valid))

            y_pred = model(X_test_norm)
            y_pred_mean = y_pred.mean()
            y_error = abs(y_pred_mean - Y_test)/Y_test * 100
            w_err = y_error[:,0]
            h_err = y_error[:,1]
            # w_err /= train_time
            # h_err /= train_time
            if ((best_h_Eavg + best_w_Eavg) > (mean(w_err) + mean(h_err))):
                best_h_Eavg = mean(h_err)
                best_w_Eavg = mean(w_err)
                best_w_err = w_err
                best_h_err = h_err
                best_layer = i
                best_dense = k
                print('Width error is: ')
                print('Max: ' + str(get_value(max(best_w_err))) + ' Mean: ' + str(get_value(best_w_Eavg)) + ' Std: ' + str(get_value(std(best_w_err)))) 
                print('Height error is: ')
                print('Max: ' + str(get_value(max(best_h_err))) + ' Mean: ' + str(get_value(best_h_Eavg)) + ' Std: ' + str(get_value(std(best_h_err)))) 
                model.save_weights('whole_PNN_model.h5')
            # w_err = tf.zeros(Y_test.shape[0])
            # h_err = tf.zeros(Y_test.shape[0])

print('')
print('#################')
print('Best error value obtained is: ')
print('Width error is: ')
print('Max: ' + str(get_value(max(best_w_err))) + ' Mean: ' + str(get_value(best_w_Eavg)) + ' Std: ' + str(get_value(std(best_w_err)))) 
print('Height error is: ')
print('Max: ' + str(get_value(max(best_h_err))) + ' Mean: ' + str(get_value(best_h_Eavg)) + ' Std: ' + str(get_value(std(best_h_err)))) 
print('With the hidden layer size: ' + str(best_layer) + '-' + str(best_dense))