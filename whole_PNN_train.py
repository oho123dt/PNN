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

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load


PNN_mode = 1
######### IMPORT DATA ###############

F_train,S_train,V_train,D_train,W_train,H_train = read_data('train.csv')
X = np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
Y = np.concatenate((W_train.reshape(-1,1),H_train.reshape(-1,1)),1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

indices = tf.range(start=0, limit=tf.shape(X_std)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices, seed = 1)

X_std = tf.gather(X_std, shuffled_indices)
Y = tf.gather(Y, shuffled_indices)

X_train_std = X_std[0:29]
Y_train = Y[0:29]
X_valid_std = X_std[30:]
Y_valid = Y[30:]

dump(scaler, 'std_scaler.bin', compress=True)

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution
act = 'relu'

######### CALL BACK AND LOSS FUNCTION ##########

model_cb=ModelCheckpoint('./PNN_model_NoScale.h5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=500,verbose=1,restore_best_weights=True)
cb = [model_cb, early_cb]

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

if PNN_mode:
    model = Sequential ([
        tfpl.DenseVariational (
            units = 10, input_shape = (input_shape,),
            make_prior_fn = prior, make_posterior_fn = posterior,
            kl_weight = 1 / X_train_std.shape[0], activation = 'sigmoid'
        ),    
        tfk.layers.Dense (
            units = 4, activation = 'relu'
        ),
        tfk.layers.Dense (
            units = tfpl.MultivariateNormalTriL.params_size(2),
        ),
        tfpl.MultivariateNormalTriL(2)
    ])
    model.compile(loss=NLL, optimizer= RMSprop(learning_rate = 0.005))              # RMSprop(learning_rate = 0.005)

else:
    model = Sequential ([
        tfk.layers.Dense (            
            units = 10, activation = 'relu', input_shape = (input_shape, ),
        ),
        Dropout(0.2),
        tfk.layers.Dense (
            units = tfpl.MixtureNormal.params_size(num_components, output_shape),
        ),
        tfpl.MixtureNormal(num_components, output_shape)
    ])
    model.compile(loss=NLL, optimizer= 'adam')              # RMSprop(learning_rate = 0.005)


history = model.fit(X_train_std,Y_train,epochs=3000,batch_size=4,verbose=1,callbacks=early_cb,shuffle=True,validation_data=(X_valid_std, Y_valid))

model.save_weights('whole_PNN_model'+str(PNN_mode)+'.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
if PNN_mode:
    plt.ylim([0,20])
else:
    plt.ylim([0,8])
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('whole_PNN_loss_' + str(PNN_mode) +'.png')