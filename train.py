#!/usr/bin/env python
# coding: utf-8

# # 1. Create model

# In[1]:


import runet as vae_util


# In[ ]:

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.keras.optimizers import Adam

import h5py
print(tf.__version__)

# In[6]:


input_shape = (128, 128, 3)



# # 2. Load data

# In[ ]:


# 
data_dir = './data_dir/'
hf_r = h5py.File(data_dir + 'train_200000.hdf5', 'r')
train_x = np.array(hf_r.get('k'))
train_y = np.array(hf_r.get('S'))
hf_r.close()


# In[ ]:

# hf_w = h5py.File(data_dir + 'test_200000.hdf5', 'r')
hf_w = h5py.File(data_dir + 'test_30400.hdf5', 'r')
test_x = np.array(hf_w.get('k'))
test_y = np.array(hf_w.get('S'))
hf_w.close()


# In[ ]:


print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('test_x shape is ', test_x.shape)
print('test_y shape is ', test_y.shape)


# In[ ]:


train_x = train_x.transpose(2, 0, 1, 3)
train_y = train_y.transpose(2, 0, 1, 3)
test_x = test_x.transpose(2, 0, 1, 3)
test_y = test_y.transpose(2, 0, 1, 3)
print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('test_x shape is ', test_x.shape)
print('test_y shape is ', test_y.shape)


# # 3. Training

# In[ ]:


# Define Loss
def vae_loss(x, t_decoded):
    '''Total loss for the plain UAE'''
    return K.mean(reconstruction_loss(x, t_decoded))

def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain UAE'''
    return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)


# In[ ]:


# Training specification
epoch = 50
train_nr = train_x.shape[0]
batch_size = 16
test_nr = 80
num_batch = int(train_nr/batch_size) 
learning_rate = 1e-4

opt = Adam(lr = learning_rate)

train_target = K.placeholder(shape=(batch_size, 128, 128, 1))
test_target = K.placeholder(shape=(test_nr, 128, 128, 1))



strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
# with strategy.scope():
vae_model = vae_util.create_vae(input_shape)
vae_model.compile(optimizer=opt, loss='mse')
rec_loss = vae_loss(vae_model.output, train_target)
total_loss = rec_loss

updates = opt.get_updates(total_loss, vae_model.trainable_weights)

iterate = K.function(vae_model.inputs + [train_target], [rec_loss], updates=updates)

eval_rec_loss = vae_loss(vae_model.output, test_target)

evaluate = K.function(vae_model.inputs + [test_target], [eval_rec_loss])

output_dir = './saved_models/'


# In[ ]:


# Train
for e in range(epoch):
    for ib in range(num_batch):
        ind0 = ib * batch_size
        n_itr = e * train_nr + ind0 + batch_size # for tensorboard output
        x_batch = train_x[ind0:ind0+batch_size, ...]
        y_batch = train_y[ind0:ind0+batch_size, ...]
        rec_loss_val = iterate([x_batch] + [y_batch])
        eval_loss_val = evaluate([test_x[:100,...]] + [test_y[:100,...]])
        
        if ib % 100 == 0:
            print('Epoch %d/%d, Batch %d/%d, Rec Loss %f' % (e+1, epoch, ib+1, num_batch, rec_loss_val[0]))
    print('Epoch %d/%d, Train Rec loss %f, Eval Rec loss %f' % (e + 1, epoch, rec_loss_val[0], eval_loss_val[0]))

    if (e+1) % 10 == 0:
        vae_model.save_weights(output_dir +  'trained_model_ep%d.h5' % ((e+1)))

vae_model.save_weights(output_dir +  'trained_model_ep%d.h5' % ((e+1)))


# In[ ]:




