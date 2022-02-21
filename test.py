# Activate TF2 behavior:
from read_csv import *
from keras.layers import Input, Add, Dense, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

train_mode = False
import matplotlib.pylab as plt
import seaborn as sns
import time

import re
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from joblib import dump, load

weights_filepath = 'custom_model.h5'

F_test,S_test,V_test,D_test,W_test,H_test = read_data('test.csv')
X_test = np.concatenate((F_test.reshape(-1,1),S_test.reshape(-1,1),V_test.reshape(-1,1),D_test.reshape(-1,1)),1)
Y_test = np.concatenate((W_test.reshape(-1,1),H_test.reshape(-1,1)),1)

sc = StandardScaler()
sc = load('fix_scaler.bin')

X_test_norm = sc.transform(X_test)

num_components = 1 # Number of components in the Gaussian Mixture
input_shape = 4
output_shape = [2] # # Shape of the distribution
act = 'relu'
model = keras.Sequential([
    keras.layers.Dense(units=10, activation=act, input_shape=(input_shape,)),
    keras.layers.Dense(units = 2)])

model.load_weights(weights_filepath)

# Infer
start = time.time()
y_pred = model(X_test_norm)

end = time.time()
print("Time runs: {}".format(end-start))

# f = open("org_noScale.txt","w")

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


x = np.arange(1,8)
label = {0:'Width',1:'Height'}
color = ['go','r^','r*-']
limit = {0:(6,12),1:(2,4)}

for i in range(2):
    plt.figure(figsize=(12,5))
    plt.plot(x, y_pred[:, i],color[0],label='Predicted mean')
    plt.plot(x, Y_test[:,i], color[1], alpha = 0.5, label='True')
    plt.ylabel(label[i])
    plt.xlabel('X')
    plt.ylim(limit[i])
    plt.legend(loc='upper right')
    plt.savefig('Fix_NS_'+str(i)+'_.png')