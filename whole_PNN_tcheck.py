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

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics as sts
from joblib import dump, load

PNN_mode = 1

weights_filepath = 'whole_PNN_model'+str(PNN_mode)+'.h5'

num_components = 3 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution

F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

sc = StandardScaler()
sc = load('std_scaler.bin')

X_test_std = sc.transform(X_test)

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
            kl_weight = 1 / X_test.shape[0], activation = 'sigmoid'
        ),
        tfk.layers.Dense (
            #units = tfpl.IndependentNormal.params_size(2),
            units = tfpl.MultivariateNormalTriL.params_size(2),
            #make_prior_fn = prior, make_posterior_fn = posterior,
            #kl_weight = 1 / X_test.shape[0]
        ),
        #tfpl.IndependentNormal(2)
        tfpl.MultivariateNormalTriL(2)
    ])
    model.compile(loss=NLL, optimizer= RMSprop(learning_rate = 0.005))              # RMSprop(learning_rate = 0.005)

else:
    model = Sequential ([
        tfk.layers.Dense (
            units = 10, input_shape = (input_shape,), activation = 'relu'
        ),
        tfk.layers.Dense (
            units = tfpl.IndependentNormal.params_size(2),
            #units = tfpl.MultivariateNormalTriL.params_size(2),
            #make_prior_fn = prior, make_posterior_fn = posterior,
            #kl_weight = 1 / X_test.shape[0]
        ),
        tfpl.IndependentNormal(2)
        #tfpl.MultivariateNormalTriL(2)
    ])
    model.compile(loss=NLL, optimizer= RMSprop(learning_rate = 0.005))              # RMSprop(learning_rate = 0.005)


model.load_weights(weights_filepath)


# Infer
y_pred = model(X_test_std)


# print('Joint plot of distribution')
# print('...')

# N = 100000
# x = y_pred.sample(N)
# x1 = x[:, 0, 0]
# x2 = x[:, 0, 1]
# sns.jointplot(x1, x2, kind = 'kde', space = 0)
# plt.xlabel('Width')
# plt.ylabel('Height')
# plt.savefig('whole_PNN_joint_plot'+str(PNN_mode)+'.png')
# print('Done plotting')

# # Calculate mean and std, covarian
mu = y_pred.mean()
sigma = y_pred.stddev()
plt.figure(figsize=(12,5))
x = np.linspace(mu[0,0] - 3*sigma[0,0], mu[0,0] + 3*sigma[0,0], 100)
plt.plot(x, stats.norm.pdf(x, mu[0,0], sigma[0,0]),label='Width distribution')
W_dist = tfd.Normal(loc = mu[0,0], scale = sigma[0,0])
plt.scatter(Y_test[:,0], W_dist.prob(Y_test[:,0]),c='red',label = 'Ground Truth')
plt.legend()
plt.xlabel('Width (mm)')
plt.savefig('whole_PNN_Width_distribution.png')

plt.figure(figsize=(12,5))
x = np.linspace(mu[0,1] - 3*sigma[0,1], mu[0,1] + 3*sigma[0,1], 100)
plt.plot(x, stats.norm.pdf(x, mu[0,1], sigma[0,1]),label='Height distribution')
W_dist = tfd.Normal(loc = mu[0,1], scale = sigma[0,1])
plt.scatter(Y_test[:,1], W_dist.prob(Y_test[:,1]),c='red',label = 'Ground Truth')
plt.legend()
plt.xlabel('Height (mm)')
plt.savefig('whole_PNN_Height_distribution.png')

# y_pred_mean = np.squeeze(y_pred_mean)
# y_pred_std = np.squeeze(y_pred_std)

# # y_pred_mean_final = scaler_1.inverse_transform(y_pred_mean)
# # y_pred_std_final = scaler_1.inverse_transform(y_pred_std)

# W_pred = y_pred_mean[:,0]
# H_pred = y_pred_mean[:,1]

# W_std = y_pred_std[:,0]
# H_std = y_pred_std[:,1]

# # x = np.linspace(-10,30,600)
# # y = np.linspace(-5,10,500)
# # X, Y = np.meshgrid(x,y)
# # pos = np.empty(X.shape + (2,))
# # pos[:, :, 0] = X; pos[:, :, 1] = Y
# # rv = multivariate_normal([y_pred_mean[0,0], y_pred_mean[0,1]], [[pow(y_pred_std[0,0],2), 0], [0, pow(y_pred_std[0,1],2)]])


# # #Make a 3D plot
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
# # ax.set_xlabel('Width')
# # ax.set_ylabel('Height')
# # ax.set_zlabel('Prob value')
# # plt.savefig('3d_distribution.png')

# f = open("whole_PNN_"+str(PNN_mode)+".txt","w")

# ######## # Ground Truth #############
# f.write("#Ground Truth\n")
# f.write("W: ")
# for i in range(W_test.shape[0]):
#   if i==W_test.shape[0] -1:
#     f.write(str(W_test[i])+"\n")
#   else:
#     f.write(str(W_test[i])+" ")
# f.write("H: ")
# for i in range(H_test.shape[0]):
#   if i==H_test.shape[0] -1:
#     f.write(str(H_test[i])+"\n")
#   else:
#     f.write(str(H_test[i])+" ")


# ######### Predict
# f.write("#Predict\n")
# f.write("W: ")
# for i in range(W_pred.shape[0]):
#   if i==W_pred.shape[0] -1:
#     f.write(str(W_pred[i])+"\n")
#   else:
#     f.write(str(W_pred[i])+" ")
# f.write("H: ")
# for i in range(H_pred.shape[0]):
#   if i==H_pred.shape[0] -1:
#     f.write(str(H_pred[i])+"\n")
#   else:
#     f.write(str(H_pred[i])+" ")

# #### STD ############
# f.write("#STD\n")
# f.write("W: ")
# for i in range(W_std.shape[0]):
#   if i==W_std.shape[0] -1:
#     f.write(str(W_std[i])+"\n")
#   else:
#     f.write(str(W_std[i])+" ")
# f.write("H: ")
# for i in range(H_std.shape[0]):
#   if i==H_std.shape[0] -1:
#     f.write(str(H_std[i])+"\n")
#   else:
#     f.write(str(H_std[i])+" ")
# f.close()


# x = np.arange(1,8)
# label = {0:'Width',1:'Height'}
# color = ['go','r^','r*-']
# limit = {0:(4,13),1:(1,6)}

# for i in range(2):
#     plt.figure(figsize=(12,5))
#     plt.errorbar(x, y_pred_mean[:, i],yerr=1.96*y_pred_std[:, i], fmt=".")
#     plt.plot(x, y_pred_mean[:, i],color[0],label='Predicted mean')
#     plt.plot(x, Y_test[:,i], color[1], alpha = 0.5, label='True')
#     #plt.yscale('symlog')
#     plt.ylabel(label[i])
#     plt.xlabel('X')
#     #plt.ylim(limit[i])
#     plt.legend(loc='upper right')
#     plt.title(label[i]+ ' prediction diagram with PNN model')
#     #plt.tight_layout()
#     plt.savefig('whole_PNN_NS_'+label[i]+str(PNN_mode)+'.png')
