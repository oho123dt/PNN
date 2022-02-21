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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

######### IMPORT DATA ###############

F_train,S_train,V_train,D_train,W_train,H_train = read_data('train.csv')

X= np.concatenate((F_train.reshape(-1,1),S_train.reshape(-1,1),V_train.reshape(-1,1),D_train.reshape(-1,1)),1)
Y = W_train.reshape(-1,1)

scaler = StandardScaler()
#X_std = scaler.fit_transform(X)
X_std = X

indices = tf.range(start=0, limit=tf.shape(X_std)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)
X_std = tf.gather(X_std, shuffled_indices)
Y = tf.gather(Y, shuffled_indices)
X_train_std = X_std[0:29]
Y_train = Y[0:29]
X_valid_std = X_std[30:]
Y_valid = Y[30:]

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [1] # # Shape of the distribution
act = 'relu'

######### CALL BACK AND LOSS FUNCTION ##########

model_cb=ModelCheckpoint('./PNN_model_NoScale.h5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=300,verbose=1, restore_best_weights=True)
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


model = Sequential ([
    tfpl.DenseVariational (
        units = 10, input_shape = (input_shape,),
        make_prior_fn = prior, make_posterior_fn = posterior,
        kl_weight = 1 / X_train_std.shape[0], kl_use_exact = True, activation="sigmoid"
    ),
    Dense(units = 2)
])
model.compile(loss= MeanSquaredError(), optimizer= RMSprop(learning_rate = 0.005))              # RMSprop(learning_rate = 0.005)

model.fit(X_train_std, Y_train,epochs=2000,batch_size=4,verbose=1,callbacks=early_cb,shuffle=True,validation_data=(X_valid_std, Y_valid))

F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')

X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

#X_test_norm = scaler.transform(X_test)
X_test_norm = X_test

# Infer
x = np.arange(1,8)
label = {0:'Width',1:'Height'}
color = ['go','r^','r*-']
limit = {0:(5,13),1:(2,5)}

W_predictions = []
H_predictions = []
H_prediction_mean = []
W_prediction_mean = []
H_err = np.empty([2,1])
W_err = np.empty([2,1])
print("start inferring")
for i in range(X_test_norm.shape[0]):
    W_predicted = []
    H_predicted = []
    for _ in range(100):
        y_pred = model(X_test_norm[i,:].reshape(1,-1))
        W_predicted.append(y_pred[0,0])
        W_predictions.append(y_pred[0,0])
        H_predicted.append(y_pred[0,1])
        H_predictions.append(y_pred[0,1])

    H_prediction_mean.append (np.mean(H_predicted))
    W_prediction_mean.append (np.mean(W_predicted))
    H_lower = np.mean(H_predicted) - np.min(H_predicted)
    W_lower = np.mean(W_predicted) - np.min(W_predicted)
    H_upper = np.max(H_predicted) - np.mean(H_predicted)
    W_upper = np.max(W_predicted) - np.mean(W_predicted)
    H_err = np.append(H_err, np.array([H_lower, H_upper]).reshape(2,-1), axis = 1)
    W_err = np.append(W_err, np.array([W_lower, W_upper]).reshape(2,-1), axis = 1)

print("finish inferring")
# draw
plt.figure(figsize=(12,5))
plt.plot(x, W_prediction_mean, color[0], label='Predicted mean')
plt.errorbar(x, W_prediction_mean , yerr = W_err[:,1:8], fmt="o")
plt.plot(x, Y_test[:,0], color[1], alpha = 0.5, label='True')
plt.ylabel(label[0])
plt.xlabel('X')
plt.ylim(limit[0])
plt.legend(loc='upper right')
plt.savefig('Width_prediction'+'.png')

plt.figure(figsize=(12,5))
plt.plot(x, H_prediction_mean, color[0], label='Predicted mean')
plt.errorbar(x, H_prediction_mean , yerr = H_err[:, 1:8], fmt="o")
plt.plot(x, Y_test[:,1], color[1], alpha = 0.5, label='True')
plt.ylabel(label[0])
plt.xlabel('X')
plt.ylim(limit[1])
plt.legend(loc='upper right')
plt.savefig('Height_prediction'+'.png')

############# BOXPLOT ###############

W_predictions = np.reshape(W_predictions, (-1,7))
plt.figure(figsize=(12,5))
plt.boxplot(W_predictions)
plt.plot(x, Y_test[:,0], color[1], label = 'Ground Truth')
plt.ylabel(label[0])
plt.ylim(limit[0])
plt.legend(loc='upper right')
plt.title('Width prediction box plot')
plt.savefig('BNN_W_boxPlt'+'.png')

H_predictions = np.reshape(H_predictions, (-1,7))
plt.figure(figsize=(12,5))
bH = plt.boxplot(H_predictions)
GND = plt.plot(x, Y_test[:,1], color[1], label = 'Ground Truth')
plt.ylabel(label[1])
plt.ylim(limit[1])
plt.legend(loc='upper right')
plt.title('Height prediction box plot')
plt.savefig('BNN_H_boxPlt'+'.png')
    
    # print(
    #         f"W_Mean: {round(W_prediction_mean, 3)}, "
    #         f"min: {round(W_prediction_min, 3)}, "
    #         f"max: {round(W_prediction_max, 3)}, "
    #         f"range: {round(W_prediction_range, 3)} - "
    #         f"Actual: {Y_test[i,0]}"
    #     )
    # print(
    #         f"H_Mean: {round(H_prediction_mean, 3)}, "
    #         f"min: {round(H_prediction_min, 3)}, "
    #         f"max: {round(H_prediction_max, 3)}, "
    #         f"range: {round(H_prediction_range, 3)} - "
    #         f"Actual: {Y_test[i,1]}"
    #     )