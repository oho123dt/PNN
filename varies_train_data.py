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

X= np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
Y = np.concatenate((W_train.reshape(-1,1),H_train.reshape(-1,1)),1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# indices = tf.range(start=0, limit=tf.shape(X_std)[0], dtype=tf.int32)
# shuffled_indices = tf.random.shuffle(indices)
# X_std = tf.gather(X_std, shuffled_indices)
# Y = tf.gather(Y, shuffled_indices)
# X_train_std = X_std[0:29]
# Y_train = Y[0:29]
# X_valid_std = X_std[30:]
# Y_valid = Y[30:]

dump(scaler, 'std_scaler.bin', compress=True)

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution
act = 'relu'

######### CALL BACK AND LOSS FUNCTION ##########

model_cb=ModelCheckpoint('./PNN_model_NoScale.h5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=300, verbose=1, restore_best_weights=True)
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

F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)
X_test_std = scaler.transform(X_test)

width_std = {'10':[], '15':[], '20':[], '25':[], '30':[]}
height_std = {'10':[], '15':[], '20':[], '25':[], '30':[]}

y_pred_mean = tf.zeros(Y_test.shape[1])
y_pred_std = tf.zeros(Y_test.shape[1])

x = np.arange(1,8)
label = {0:'Width',1:'Height'}
color = ['go','r^','r*-']
limit = {0:(6,12),1:(2,4)}

train_time = 1

for i in range(5):
    final = 10 + 5*i - 1
    for j in range(train_time):
        print('Training with '+str(final+1) +' train data time no. '+str(j+1))
        indices = tf.range(start=0, limit=tf.shape(X_std)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        X_std = tf.gather(X_std, shuffled_indices)
        Y = tf.gather(Y, shuffled_indices)
        X_train_std = X_std[0:29]
        Y_train = Y[0:29]
        X_valid_std = X_std[30:]
        Y_valid = Y[30:]
        model = Sequential ([
            tfpl.DenseVariational (
                units = 10, input_shape = (input_shape,),
                make_prior_fn = prior, make_posterior_fn = posterior,
                kl_weight = 1 / (final+1), activation = 'sigmoid'
            ),    
            tfk.layers.Dense (
                units = 4, activation = 'relu'
            ),
            tfk.layers.Dense (
                units = tfpl.MultivariateNormalTriL.params_size(2),
            ),
            tfpl.MultivariateNormalTriL(2)
        ])
        model.compile(loss=NLL, optimizer= RMSprop(learning_rate = 0.005))
        model.fit(X_train_std[0:final],Y_train[0:final],epochs=2000,batch_size=4,verbose=0,callbacks=early_cb,shuffle=True,validation_data=(X_valid_std, Y_valid))
        y_pred = model(X_test_std)
        y_pred_mean += y_pred.mean()
        y_pred_std += y_pred.stddev()
    y_pred_mean = np.squeeze(y_pred_mean / train_time)
    y_pred_std = np.squeeze(y_pred_std / train_time)
    for i in range(2):
        plt.figure(figsize=(12,5))
        plt.errorbar(x, y_pred_mean[:, i],yerr=1.96*y_pred_std[:, i], fmt=".")
        plt.plot(x, y_pred_mean[:, i],color[0],label='Predicted mean')
        plt.plot(x, Y_test[:,i], color[1], alpha = 0.5, label='True')
        plt.ylabel(label[i])
        plt.xlabel('X')
        plt.ylim(limit[i])
        plt.legend(loc='upper right')
        plt.title(label[i]+ ' prediction diagram with PNN model')
        plt.savefig('train_'+label[i]+'_'+str(final+1)+'.png')
        plt.close()
    width_std[str(final+1)] = y_pred_std[:,0]
    height_std[str(final+1)] = y_pred_std[:,1]
    y_pred_mean = tf.zeros(y_pred_mean.shape)
    y_pred_std = tf.zeros(y_pred_std.shape)

X = list(width_std.keys())
y_axis = list(width_std.values())
y1_axis = list(height_std.values())
x_axis = np.arange(len(X))

fig, axs = plt.subplots(3, 3, figsize=(12,7))

for i in range(7):
    axs[int(i/3), i%3].set_title('Test case '+str(i))
    axs[int(i/3), i%3].bar(x_axis-0.2, [y_axis[0][i],y_axis[1][i],y_axis[2][i],y_axis[3][i],y_axis[4][i]], width = 0.4, label = 'Width')
    axs[int(i/3), i%3].bar(x_axis+0.2, [y1_axis[0][i],y1_axis[1][i],y1_axis[2][i],y1_axis[3][i],y1_axis[4][i]], width = 0.4, label = 'Height')
    axs[int(i/3), i%3].legend()
    axs[int(i/3), i%3].set_xticks(x_axis)
    axs[int(i/3), i%3].set_xticklabels(X)
    
for ax in axs.flat:
    ax.set(xlabel='Train size')

fig.tight_layout()
plt.savefig('train_size_varies.png')
plt.close()

    
