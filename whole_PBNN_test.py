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
from tensorflow.keras.backend import mean, std, max, get_value
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import statistics as sts
from joblib import dump, load
from mpl_toolkits.mplot3d import Axes3D

weights_filepath = 'whole_PNN_model.h5'

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution

F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

sc = MinMaxScaler(feature_range = (-1,1))
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

model = Sequential ([
    tfpl.DenseVariational (
        units = 9,  input_shape = (input_shape,),
        make_prior_fn = prior, make_posterior_fn = posterior,
        kl_weight = 1 / 25, activation = 'sigmoid'
    ),  
    tfk.layers.Dense (
        units = 6, activation = 'relu'
    ),              
    tfk.layers.Dense (
        units = tfpl.MultivariateNormalTriL.params_size(2),
    ),
    tfpl.MultivariateNormalTriL(2)
])
model.load_weights(weights_filepath)

# Infer
test_time = 5
w_err = tf.zeros(Y_test.shape[0])
h_err = tf.zeros(Y_test.shape[0])
y_error_avg = tf.zeros(Y_test.shape[1])

for i in range(test_time):
    y_pred = model(X_test_std)
    y_pred_mean = y_pred.mean()
    y_error = abs(y_pred_mean - Y_test)/Y_test * 100
    w_err += y_error[:,0]
    h_err += y_error[:,1]
    print("#############")
    print("Test no. " + str(i+1))
    print('Width error is: ')
    print('Max: ' + str(get_value(max(y_error[:,0]))) + ' Mean: ' + str(get_value(mean(y_error[:,0]))) + ' Std: ' + str(get_value(std(y_error[:,0])))) 
    print('Height error is: ')
    print('Max: ' + str(get_value(max(y_error[:,1]))) + ' Mean: ' + str(get_value(mean(y_error[:,1]))) + ' Std: ' + str(get_value(std(y_error[:,1])))) 
    
w_err /= test_time
h_err /= test_time
# print("#############")
# print(w_err)
# print(h_err)
print("#############")
print("Average value among tests: ")
print('Width error is: ')
print('Max: ' + str(get_value(max(w_err))) + ' Mean: ' + str(get_value(mean(w_err))) + ' Std: ' + str(get_value(std(w_err)))) 
print('Height error is: ')
print('Max: ' + str(get_value(max(h_err))) + ' Mean: ' + str(get_value(mean(h_err))) + ' Std: ' + str(get_value(std(h_err)))) 

#### JOINT PLOT DISTRIBUTION ####

# print('Joint plot of distribution')
# print('...')

# N = 10000
# x = y_pred.sample(N)
# x1 = x[:, 0, 0]
# x2 = x[:, 0, 1]
# sns.jointplot(x1, x2, kind = 'kde', space = 0)
# plt.xlabel('Width')
# plt.ylabel('Height')
# plt.savefig('whole_PNN_joint_plot'+str(PNN_mode)+'.png')
# print('Done plotting')
#################################


# Calculate mean and std, covarian
# y_pred_mean = y_pred.mean()
# y_pred_std = y_pred.stddev()
# y_error = abs(y_pred_mean - Y_test)/Y_test * 100
# print(y_error)
# print(sum(y_error) / Y_test.shape[0])

# y_pred_mean = np.squeeze(y_pred_mean)
# y_pred_std = np.squeeze(y_pred_std)

# W_pred = y_pred_mean[:,0]
# H_pred = y_pred_mean[:,1]

# W_std = y_pred_std[:,0]
# H_std = y_pred_std[:,1]

#Make a 3D plot
# x = np.linspace(6,11,500)
# y = np.linspace(2,4,500)
# X, Y = np.meshgrid(x,y)
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X; pos[:, :, 1] = Y

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#rv = multivariate_normal(y_pred.mean()[0] , y_pred.covariance()[0])
# ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
# ax.set_xlabel('Width')
# ax.set_ylabel('Height')
# ax.set_zlabel('Prob value')
# plt.show()

x = np.arange(1, Y_test.shape[0]+1)
label = {0:'Width',1:'Height'}
color = ['go','r^','r*-']
limit = {0:(4,13),1:(1,6)}

########### 2D PLOT #############

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
#     plt.savefig('whole_PNN_test_'+label[i]+str(PNN_mode)+'.png')


# for i in range(2):
#   plt.figure(figsize=(12,5))
#   plt.bar(x, y_error[:,i] ,color = 'maroon' ,width = 0.4, label='Error percent')
#   plt.ylabel('error percent')
#   plt.xlabel('X')
#   plt.legend(loc='upper right')
#   plt.title(label[i]+ ' error percent with PNN model')
#   plt.savefig('whole_PNN_test_error_'+label[i]+'.png')

# loop_time = 100
# y_err_sum = tf.zeros(Y_test.shape[1])

# for i in range (loop_time):
#   y_pred = model(X_test_std)
#   y_pred_mean = y_pred.mean()
#   y_error = abs(y_pred_mean - Y_test)/Y_test * 100
#   y_err_sum += y_error

# y_err_sum /= loop_time
# print ('average error percent: ')
# print (y_err_sum)
# print (sum(y_err_sum) / Y_test.shape[0])

# for i in range(2):
#   plt.figure(figsize=(12,5))
#   plt.bar(x, y_err_sum[:,i] ,color = 'maroon' ,width = 0.4, label='Error percent')
#   plt.ylabel('error percent')
#   plt.xlabel('X')
#   plt.legend(loc='upper right')
#   plt.title(label[i]+ ' error percent with PNN model')
#   plt.savefig('average_error_'+label[i]+'.png')