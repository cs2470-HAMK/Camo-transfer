from data import *
from transformations import *
from GAN import *

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
landscape_path = 'data/Landscape_Images' 
camo_path = 'data/camo_subset_raw'

save_image_path = 'results'
# -

print(landscape_path, camo_path, save_image_path)

# ### Reading in Data:

BUFFER_SIZE = 1000
BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256

### Data:
LIMIT = 15 # num of landscape and camo images
train_landscape, test_landscape = data_preprocessor(landscape_path, limit=LIMIT)
train_camo, test_camo = data_preprocessor(camo_path, limit=LIMIT)

train_landscapes = train_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_landscapes = test_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_camos = train_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_camos = test_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ### Benchmark Images:

sample_landscape = next(iter(train_landscapes))
sample_camo = next(iter(train_camos))

# ### Model Fitting:

### Model:
OUTPUT_CHANNELS = 3


def fit_hyperparameter_setting(weights, n_epochs=40):
    """
    Parameters:
        weights : {
            'reconst_weight': ..., # single scalar value
            'ls_disc_weight': ...,
            'color_d_weight': ...
        }
    
    Returns: @TODO
        tensorboardX output graphs : ...
        Saved camo_GAN model? :
        Saved training progress images
    """
    camo_GAN = CamoGAN(OUTPUT_CHANNELS, weights['reconst_weight'], weights['ls_disc_weight'], weights['color_d_weight'])
    
    for epoch in range(n_epochs):
      start = time.time()

      n = 0
      for image_x, image_y in tf.data.Dataset.zip((train_landscapes, train_camos)):
        camo_GAN.train_step(image_x, image_y)
        if n % 10 == 0:
          print ('.', end='')
        n += 1

      clear_output(wait=True)
      # Using a consistent image (sample_horse) so that the progress of the model
      # is clearly visible.
      camo_GAN.generate_images(sample_landscape)
      camo_GAN.generate_patched_landscape(sample_landscape)

      if (epoch) % 5 == 0:
            # @TODO: checkpoint manager
    #     ckpt_save_path = ckpt_manager.save()
    #     print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
    #                                                          ckpt_save_path))

        # Save generated results:
        print("SAVING RESULTS")
        for i, test_ls in enumerate(test_landscapes.take(10)):
          # Naming convention: hyperparameters_epoch_i.png
          fc_pl_filename = f'{save_image_path}/reconst_{camo_GAN.reconst_weight}_lsdisc_{camo_GAN.ls_disc_weight}_color_{camo_GAN.color_d_weight}_fc_pl_epoch_{epoch+1}_{i}.png'

          camo_GAN.save_generated_results(test_ls, fc_pl_filename)

      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))


# ### Hyperparameter search:

reconst_weights = [0.1, 1, 2, 5]
ls_disc_weights = [0, 1, 5]
color_d_weights = [0, 1, 5, 10]

for reconst_w in reconst_weights:
    for ls_disc_w in ls_disc_weights:
        for color_d_w in color_d_weights:
            weights = {
                'reconst_weight': reconst_w, 
                'ls_disc_weight': ls_disc_w,
                'color_d_weight': color_d_w
            }
            
            fit_hyperparameter_setting(weights)

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
