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

landscape_path = 'data/Landscape_Images' 
camo_path = 'data/camo_subset_raw'

print(landscape_path, camo_path)

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

### Data:
LIMIT = 100
train_landscape, test_landscape = data_preprocessor(landscape_path, limit=LIMIT)
train_camo, test_camo = data_preprocessor(camo_path, limit=LIMIT)

train_landscapes = train_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_landscapes = test_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_camos = train_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_camos = test_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_landscape = next(iter(train_landscapes))
sample_camo = next(iter(train_camos))

### Model:
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

## Optimizer:
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


### Checkpoints:
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 40
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_landscapes, train_camos)):
    train_step(image_x, image_y,
               generator_g, discriminator_x, generator_g_optimizer, discriminator_x_optimizer, 
               generator_f, discriminator_y, generator_f_optimizer, discriminator_y_optimizer)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_landscape)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

# +

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# -

"/job:localhost/replica:0/task:0/device:GPU:0"

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

gpus = tf.config.list_logical_devices('GPU')
print(gpus)

# +
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# -

get_available_gpus()


