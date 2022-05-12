from data import *
from transformations import *
from color_distance import *
# from CamoGAN import *

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


benchmark_path = 'data/Benchmark_Images'
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
LIMIT = 25 # num of landscape and camo images
train_landscape, test_landscape = data_preprocessor(landscape_path, limit=LIMIT)
train_camo, test_camo = data_preprocessor(camo_path, limit=LIMIT)

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

benchmark_landscapes, _ = data_preprocessor(benchmark_path, train_split=1.0, shuffle=False, limit=None)

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
model_name = names[4] # 'patched_dewatermarked_colordist_deskied'

### Model:
OUTPUT_CHANNELS = 3

params = {
    'reconst': 0.5,
    'ls_disc': 1, 
    'color_d': 5 
}

# ### Tensorboard Logger:

# %load_ext tensorboard

# +
if not os.path.exists('tensorboard_logs'):
    os.makedirs('tensorboard_logs')

tb_log_dir = 'tensorboard_logs/gradient_tape'
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


# # 'Model':

# ### Image Generation/Saving:

# +
def generate_images(test_input):
    prediction = generator_g(test_input)

    display_list = [test_input[0], prediction[0]]
    title = ['Input Landscape', 'Predicted Camouflage']

    # @TODO: Refactor into visualization.py
    plt.figure(figsize=(12, 12))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def generate_patched_landscape(landscape):
    fake_camo = generator_g(landscape)
    patched_landscape = cutout_and_replace(landscape, fake_camo, cutout_size=64)

    display_list = [landscape[0], patched_landscape[0]]
    title = ['Input Landscape', 'Patched Landscape']

    # @TODO: Refactor into visualization.py
    plt.figure(figsize=(12, 12))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def save_generated_results(landscape, filename):
    fake_camo = generator_g(landscape)
    patched_landscape = cutout_and_replace(landscape, fake_camo, cutout_size=64)

    display_list = [fake_camo[0], patched_landscape[0]]
    title = ['Generated Camo', 'Patched Landscape']

    # @TODO: Refactor into visualization.py
    plt.figure(figsize=(12, 12))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(filename)

def save_benchmark_results(epoch, benchmark_landscapes, save_image_path, num_benchmarks=10):
    plt.figure(figsize=(20, 20))
    benchmark_camo_results_path = f'{save_image_path}/epoch_{epoch}_camo.png'
    for i, benchmark in enumerate(benchmark_landscapes.take(num_benchmarks)):
        fake_camo = generator_g(benchmark)

        plt.subplot(1, num_benchmarks, i+1)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(fake_camo[0] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(benchmark_camo_results_path)

    plt.figure(figsize=(20, 20))
    benchmark_patched_results_path = f'{save_image_path}/epoch_{epoch}_camo.png'
    for i, benchmark in enumerate(benchmark_landscapes.take(num_benchmarks)):
        fake_camo = generator_g(benchmark)
        patched_landscape = cutout_and_replace(benchmark, fake_camo, cutout_size=64)

        plt.subplot(1, num_benchmarks, i+1)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(patched_landscape[0] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(benchmark_patched_results_path)


# -

# ### Train step:

@tf.function
def train_step(real_x, real_y, step_i, reconst_weight, ls_disc_weight, color_d_weight, log_freq=20):
    """
    log_freq - after how many images is the loss logged
    """
    # y - camo
    # x - landscape
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True) # generated camouflage

        fake_x = cutout_and_replace(real_x, fake_y, cutout_size=64) # cutout/replaced landscape
        cycled_y = generator_g(fake_x, training=True) # Might not be necessary for camo GAN; how close does the generator get to producing the same camouflage from landscapes patched by camouflage?

        # same_x and same_y are used for identity loss.
        # same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True) # Can generator G produce the same camouflage given itself? (identity)

        disc_real_x = discriminator_x(real_x, training=True) # Is real landscape a landscape?
        disc_real_y = discriminator_y(real_y, training=True) # Is real camouflage a camouflage?

        disc_fake_x = discriminator_x(fake_x, training=True) # Tests: is cutout/replaced landscape a landscape?
        disc_fake_y = discriminator_y(fake_y, training=True) # Tests: is generated camouflage a camouflage?

        # Color distance penalty
        color_dist = 0
        if color_d_weight != 0:
            color_dist = calc_color_distance(real_x, fake_y)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)+ ls_disc_weight*generator_loss(disc_fake_x) + color_d_weight*color_dist# Want to fool both discriminators 

        total_cycle_loss = calc_cycle_loss(real_y, cycled_y, reconst_weight)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y, reconst_weight)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        if train_summary_writer and (step_i+1) % log_freq == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=step_i)

                tf.summary.scalar('disc_x_loss', disc_x_loss, step=step_i)
                tf.summary.scalar('disc_y_loss', disc_y_loss, step=step_i)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


# ### Model Fitting:

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
    for epoch in range(n_epochs):
      start = time.time()

    n = 0
    data_size = len(train_landscapes)
    for i, (image_x, image_y) in enumerate(tf.data.Dataset.zip((train_landscapes, train_camos))):
      step = epoch*data_size + i
      train_step(image_x, image_y, step, 
                 weights['reconst'], weights['ls_disc'], weights['color_d'],
                log_freq=1)
      if n % 10 == 0:
        print ('.', end='')
      n += 1

      clear_output(wait=True)
      # Using a consistent image (sample_horse) so that the progress of the model
      # is clearly visible.
      generate_images(sample_landscape)
      generate_patched_landscape(sample_landscape)

      if (epoch + 1) % 1 == 0: # 5
        # Save generated results:
        print("Saving results")
        save_benchmark_results(epoch+1, benchmark_landscapes, benchmark_images_log_dir)

      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))


# ### Model Fitting:

NUM_EPOCHS = 40

# +
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

# +
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# -

fit_hyperparameter_setting(weights=params, n_epochs=NUM_EPOCHS)





# # @TODO: Not implemented: paths don't update

# ### Hyperparameter search:

# +
# reconst_weights = [0.5, 1, 5]
# ls_disc_weights = [1, 5] # add 0
# color_d_weights = [1, 5, 10] # add 0

# +
# for reconst_w in reconst_weights:
#     for ls_disc_w in ls_disc_weights:
#         for color_d_w in color_d_weights:
#             weights = {
#                 'reconst_weight': reconst_w, 
#                 'ls_disc_weight': ls_disc_w,
#                 'color_d_weight': color_d_w
#             }
            
#             fit_hyperparameter_setting(weights, n_epochs=100)
# -

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
