from data import *
from transformations import *
from CamoGAN import *

# images
import glob
from PIL import Image

# +
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

import tensorflow_datasets as tfds # pip install tensorflow_datasets, etc.
from tensorflow_examples.models.pix2pix import pix2pix # pip install -q git+https://github.com/tensorflow/examples.git
# -

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

# ### Data and Output Paths:

# +
# landscape_path = 'data/Landscape_Images' 
# camo_path = 'data/camo_subset_raw'


# benchmark_path = 'data/Benchmark_Images'

research_path = 'C:/Users/mtapi/OneDrive/Documents/Brown/cs2470/Final_Project/research'
landscape_path = research_path +  '/NoSky_Landscape' # '/Landscape_Images' 
camo_path = research_path + '/camo_processed' # '/camo_subset_raw' 

benchmark_path = research_path + '/Benchmark_Images'

# -

print(f'Landscape images read from: {landscape_path};\n\
Camouflage images read from: {camo_path};\n\
Benchmark landscape images read from: {benchmark_path}')

# ### Reading in Data:

BUFFER_SIZE = 1000
BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256

### Data:
LIMIT = 100 # num of landscape and camo images
train_landscape, test_landscape = data_preprocessor(landscape_path, limit=LIMIT)
train_camo, test_camo = data_preprocessor(camo_path, image_type='*.png', limit=LIMIT)

train_landscapes = train_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_landscapes = test_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_camos = train_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_camos = test_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# +
num_camos = 10
plt.figure(figsize=(20, 20))
for i, camo in enumerate(train_camos.take(num_camos)):
    plt.subplot(1, num_camos, i+1)
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(camo[0] * 0.5 + 0.5)
    plt.axis('off')

plt.show()

# +
num_landscapes = 10
plt.figure(figsize=(20, 20))
for i, landscape in enumerate(train_landscapes.take(num_landscapes)):
    plt.subplot(1, num_landscapes, i+1)
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(landscape[0] * 0.5 + 0.5)
    plt.axis('off')

plt.show()
# -

# ### Benchmark Images:

benchmark_landscapes, _ = data_preprocessor(benchmark_path, sort=True, train_split=1.0, shuffle=False, limit=None)

benchmark_landscapes = benchmark_landscapes.cache().batch(BATCH_SIZE)

# +
num_benchmarks = 10
plt.figure(figsize=(20, 20))
for i, benchmark in enumerate(benchmark_landscapes.take(num_benchmarks)):
    plt.subplot(1, num_benchmarks, i+1)
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(benchmark[0] * 0.5 + 0.5)
    plt.axis('off')

plt.show()

# +
sample_landscape = next(iter(train_landscapes))
sample_camo = next(iter(train_camos))

plt.subplot(121)
plt.title('Sample Landscape')
plt.imshow(sample_landscape[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Sample Camo')
plt.imshow(sample_camo[0] * 0.5 + 0.5)
# -

# ## Model Name: Important for logging

names = ['cyclegan', 'patched', 'patched_dewatermarked', 'patched_dewatermarked_colordist', 'patched_dewatermarked_colordist_deskied']
model_name = names[3]
model_name = 'patched_dewatermarked_deskied'

### Model:
OUTPUT_CHANNELS = 3

params = {
    'reconst': 0.2,
    'ls_disc': 5, 
    'color_d': 0
}

# ### Tensorboard Logger:

# %load_ext tensorboard

# +
working_dir = 'C:/Users/mtapi/OneDrive/Documents/Brown/cs2470/Final_Project/Camo-transfer'
tensorboard_dir = working_dir + '/tensorboard_logs'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

tb_log_dir = tensorboard_dir + '/gradient_tape'
if not os.path.exists(tb_log_dir):
    os.makedirs(tb_log_dir)
    
model_log_dir = f'{tb_log_dir}/{model_name}'
if not os.path.exists(model_log_dir):
    os.makedirs(model_log_dir)
    
param_str = "_".join([f'{param}_{val}' for param, val in params.items()])
model_param_log_dir = f'{model_log_dir}/{param_str}'
if not os.path.exists(model_param_log_dir):
    os.makedirs(model_param_log_dir)
    
train_log_dir = model_param_log_dir

print(f'Tensorboard logs written to: {train_log_dir}')
# -

train_summary_writer = tf.summary.create_file_writer(train_log_dir)

benchmark_images_log_dir = f'{train_log_dir}/progress_images'
if not os.path.exists(benchmark_images_log_dir):
    os.makedirs(benchmark_images_log_dir)
print(f'Images written to: {benchmark_images_log_dir}')

# ### Model Fitting:

# +
n_epochs=40
camo_GAN = CamoGAN(OUTPUT_CHANNELS, 
                       params['reconst'], params['ls_disc'], params['color_d'],
                      train_summary_writer)
    
for epoch in range(n_epochs):
    start = time.time()

    data_size = len(train_landscapes)
    for i, (image_x, image_y) in enumerate(tf.data.Dataset.zip((train_landscapes, train_camos))):
        step = epoch*data_size + i
        camo_GAN.train_step(image_x, image_y, step)
        if i % 10 == 0:
            print ('.', end='')

    # clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    """ Remove for local runs
    camo_GAN.generate_images(sample_landscape)
    camo_GAN.generate_patched_landscape(sample_landscape)
    """

    if (epoch + 1) % 5 == 0:
        # Save generated results:
        print("Saving results")
        camo_GAN.save_benchmark_results(epoch+1, benchmark_landscapes, benchmark_images_log_dir)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


# -

'''
def fit_hyperparameter_setting(weights, n_epochs=40):
    """
    Parameters:
        weights : {
            'reconst': ..., # single scalar value
            'ls_disc': ...,
            'color_d': ...
        }
    
    Returns: @TODO
        tensorboardX output graphs : ...
        Saved camo_GAN model? :
        Saved training progress images
    """
    camo_GAN = CamoGAN(OUTPUT_CHANNELS, 
                       weights['reconst'], weights['ls_disc'], weights['color_d'],
                      train_summary_writer)
    
    for epoch in range(n_epochs):
        start = time.time()

        n = 0
        data_size = len(train_landscapes)
        for i, (image_x, image_y) in enumerate(tf.data.Dataset.zip((train_landscapes, train_camos))):
            step = epoch*data_size + i
            camo_GAN.train_step(image_x, image_y, step)
            if n % 10 == 0:
                print ('.', end='')
                n += 1

        clear_output(wait=True)
        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        camo_GAN.generate_images(sample_landscape)
        camo_GAN.generate_patched_landscape(sample_landscape)

    if (epoch + 1) % 5 == 0:
        # Save generated results:
        print("Saving results")
        camo_GAN.save_benchmark_results(epoch+1, benchmark_landscapes, benchmark_images_log_dir)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))


fit_hyperparameter_setting(params, n_epochs=100)
'''


# # @TODO: not implemented

# ### Hyperparameter search:

# reconst_weights = [0.5, 1, 5]
# ls_disc_weights = [1, 5] # add 0
# color_d_weights = [1, 5, 10] # add 0

# for reconst_w in reconst_weights:
#     for ls_disc_w in ls_disc_weights:
#         for color_d_w in color_d_weights:
#             weights = {
#                 'reconst_weight': reconst_w, 
#                 'ls_disc_weight': ls_disc_w,
#                 'color_d_weight': color_d_w
#             }

#             fit_hyperparameter_setting(weights, n_epochs=100)

# ### Testing GPU:

# +
# from tensorflow.python.client import device_lib

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

# +
# get_available_gpus()

# +
# gpus = tf.config.list_logical_devices('GPU')
# print(gpus)
