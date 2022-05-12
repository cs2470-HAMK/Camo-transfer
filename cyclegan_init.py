from data import *
from transformations import *
from CycleGAN import *

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

# +
landscape_path = 'data/Landscape_Images' 
camo_path = 'data/camo_subset_raw'

benchmark_path = 'data/Benchmark_Images'
# -

print(f'Landscape images read from: {landscape_path};\n\
Camouflage images read from: {camo_path};\n\
Benchmark landscape images read from: {benchmark_path}')

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# +
### Data:
LIMIT = 100
train_landscape, test_landscape = data_preprocessor(landscape_path, limit=LIMIT)
train_camo, test_camo = data_preprocessor(camo_path, limit=LIMIT)

benchmark_landscapes, _ = data_preprocessor(benchmark_path, shuffle=False, limit=None)

# +
train_landscapes = train_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_landscapes = test_landscape.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_camos = train_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_camos = test_camo.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

benchmark_landscapes = benchmark_landscapes.cache().batch(BATCH_SIZE)

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

### Model:
OUTPUT_CHANNELS = 3

# ### Model Name: Important for Logging:

model_name = 'cyclegan' # 'patched'; # patched_dewatermarked'; # 'patched_dewatermarked_colordist'; # 'patched_dewatermarked_colordist_deskied'

# ### Tensorboard Logger:

# %load_ext tensorboard

reconst_weight=10

# +
if not os.path.exists('tensorboard_logs'):
    os.makedirs('tensorboard_logs')

tb_log_dir = 'tensorboard_logs/gradient_tape'
if not os.path.exists(tb_log_dir):
    os.makedirs(tb_log_dir)
    
model_log_dir = f'{tb_log_dir}/{model_name}'
if not os.path.exists(model_log_dir):
    os.makedirs(model_log_dir)
    
params = {
    'reconst': 5
}
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

cycle_GAN = CycleGAN(OUTPUT_CHANNELS, reconst_weight=reconst_weight, train_summary_writer=train_summary_writer)

# +
EPOCHS = 40
data_size = len(train_landscapes)

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for i, (image_x, image_y) in enumerate(tf.data.Dataset.zip((train_landscapes, train_camos))):
    step = epoch*data_size + i
    cycle_GAN.train_step(image_x, image_y, step, reconst_weight=reconst_weight)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  cycle_GAN.generate_images(sample_landscape)

  if (epoch + 1) % 2 == 0:
    
    cycle_GAN.save_benchmark_results(epoch+1, benchmark_landscapes, benchmark_images_log_dir)

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

  if epoch > 17:
    break
# -

# %tensorboard --logdir tensorboard_logs


