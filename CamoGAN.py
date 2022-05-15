# +
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

from tensorflow_examples.models.pix2pix import pix2pix # pip install -q git+https://github.com/tensorflow/examples.git

from transformations import *


# -

class CamoGAN(tf.keras.Model):

    '''
    Core class to administrate both the generator and discriminator
    '''

    def __init__(self, output_channels, reconst_weight=1, ls_disc_weight=1, color_d_weight=1, train_summary_writer=None):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super().__init__()
        self.train_summary_writer = train_summary_writer
        
        self.reconst_weight = reconst_weight
        self.ls_disc_weight = ls_disc_weight
        self.color_d_weight = color_d_weight
        
        self.output_channels = output_channels
        self.generator_g = pix2pix.unet_generator(output_channels, norm_type='instancenorm')
        
        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
        
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.model_name = self.get_name()
        
        
    def get_name(self):
        return f"reconst_{self.reconst_weight}_ls_disc_{self.ls_disc_weight}_color_d_{self.color_d_weight}"
    
    
    def generate_images(self, test_input):
        prediction = self.generator_g(test_input)

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
    
    
    def generate_patched_landscape(self, landscape):
        fake_camo = self.generator_g(landscape)
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
        
        
    def save_generated_results(self, landscape, filename):
        fake_camo = self.generator_g(landscape)
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
        
    def save_benchmark_results(self, epoch, benchmark_landscapes, save_image_path, num_benchmarks=10):
        plt.figure(figsize=(20, 20))
        benchmark_camo_results_path = f'{save_image_path}/epoch_{epoch}_camo.png'
        for i, benchmark in enumerate(benchmark_landscapes.take(num_benchmarks)):
            fake_camo = self.generator_g(benchmark)
            
            plt.subplot(1, num_benchmarks, i+1)
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(fake_camo[0] * 0.5 + 0.5)
            plt.axis('off')
        
        # plt.close(fig)
        plt.savefig(benchmark_camo_results_path)
        print("Saved at: ", benchmark_camo_results_path)

        # fig = plt.figure(figsize=(20, 20))
        benchmark_patched_results_path = f'{save_image_path}/epoch_{epoch}_patched.png'
        for i, benchmark in enumerate(benchmark_landscapes.take(num_benchmarks)):
            fake_camo = self.generator_g(benchmark)
            patched_landscape = cutout_and_replace(benchmark, fake_camo, cutout_size=64)

            plt.subplot(1, num_benchmarks, i+1)
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(patched_landscape[0] * 0.5 + 0.5)
            plt.axis('off')

        # plt.close(fig)
        plt.savefig(benchmark_patched_results_path)
        
        
    def train_step(self, real_x, real_y, step_i, log_freq=20):
        """
        log_freq - after how many images is the loss logged
        """
        # y - camo
        # x - landscape
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_g(real_x, training=True) # generated camouflage

            fake_x = cutout_and_replace(real_x, fake_y, cutout_size=64) # cutout/replaced landscape
            cycled_y = self.generator_g(fake_x, training=True) # Might not be necessary for camo GAN; how close does the generator get to producing the same camouflage from landscapes patched by camouflage?

            # same_x and same_y are used for identity loss.
            # same_x = generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True) # Can generator G produce the same camouflage given itself? (identity)

            disc_real_x = self.discriminator_x(real_x, training=True) # Is real landscape a landscape?
            disc_real_y = self.discriminator_y(real_y, training=True) # Is real camouflage a camouflage?

            disc_fake_x = self.discriminator_x(fake_x, training=True) # Tests: is cutout/replaced landscape a landscape?
            disc_fake_y = self.discriminator_y(fake_y, training=True) # Tests: is generated camouflage a camouflage?

            # Color distance penalty
            color_dist = 0
            if self.color_d_weight != 0:
                color_dist = calc_color_distance(real_x, fake_y)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)+ self.ls_disc_weight*generator_loss(disc_fake_x) + self.color_d_weight*color_dist# Want to fool both discriminators 

            total_cycle_loss = calc_cycle_loss(real_y, cycled_y, self.reconst_weight)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y, self.reconst_weight)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
            
            if self.train_summary_writer and (step_i+1) % log_freq == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=step_i)

                    tf.summary.scalar('disc_x_loss', disc_x_loss, step=step_i)
                    tf.summary.scalar('disc_y_loss', disc_y_loss, step=step_i)

                    if self.color_d_weight != 0:
                        tf.summary.scalar('color_distance', color_dist, step=step_i)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            self.generator_g.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                self.generator_g.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    self.discriminator_y.trainable_variables))


    '''@TODO: bad shapes for tensorboard logger
    def log_benchmark_results(self, benchmark_landscapes, step_i, num_results=10):
        camo_list = []
        patched_list = []
        for i, benchmark in enumerate(benchmark_landscapes.take(num_results)):
            print("In benchmark_landscapes")
            fake_camo = self.generator_g(benchmark)
            patched_landscape = cutout_and_replace(benchmark, fake_camo, cutout_size=64)

            camo_list.append(fake_camo[0].numpy())
            patched_list.append(patched_landscape[0].numpy())
            print(np.array(camo_list).shape)
            
        camo_images = np.array(camo_list)
        patched_images = np.array(patched_list)
        print("Writing in log_bench_mark_results")
        with self.train_summary_writer.as_default():
            camo_images = np.reshape(camo_images[0:10], (-1, 256, 256, 1))
            patched_images = np.reshape(patched_images[0:10], (-1, 256, 256, 1))
            tf.summary.image("10 camo examples", camo_images, max_outputs=10, step=step_i)
            tf.summary.image("10 patched examples", patched_images, max_outputs=10, step=step_i)
    '''
        
    '''
    New calling API for the model
    '''
    
    def sample_z(self, num_samples, **kwargs):
        '''generates a z realization from the z sampler'''
#         return self.z_sampler([num_samples, *self.z_dims[1:]], safe=True)
        pass
    
    def discriminate(self, inputs, **kwargs):
        '''predict whether input input is a real entry from the true dataset'''
#         return self.dis_model(inputs, **kwargs)
        pass

    def generate(self, z, **kwargs):
        '''generates an output based on a specific z realization'''
#         return self.gen_model(z, **kwargs)
        pass

    '''
    keras.Model will handle all components defined in its end-to-end architecture.  
    As such, we can link the generator and discriminator within the GAN by:
     - Chaining the generator and discriminator together in the call function
     - Building the GAN in the way that facilitates this call function. 
    '''
    def call(self, inputs, **kwargs):
#         b_size = tf.shape(inputs)[0]
#         ## TODO: Implement call process per instructions
#         z_samp = self.sample_z(b_size) # tf.constant([0.])   ## Generate a z sample
#         g_samp = self.generate(z_samp)   ## Generate an x-like image
#         disc_input = g_samp # tf.concat([g_samp, inputs], axis=0)
#         d_samp = self.discriminate( g_samp) # tf.constant([0.])   ## Predict whether x-like is real
#         print(f'Z( ) Shape = {z_samp.shape}')
#         print(f'G(z) Shape = {g_samp.shape}')
#         print(f'D(x) Shape = {d_samp.shape}\n')
#         return d_samp
        pass

    def build(self, input_shape, **kwargs):
        super().build(input_shape=self.z_dims, **kwargs)



loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5


def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, loss_weight=10):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return loss_weight * loss1


def identity_loss(real_image, same_image, loss_wieght=10):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return loss_wieght * 0.5 * loss



def palette_perc(img, clusters: int, dim=(25, 15), verbose=False):
    # img = cv2.imread(img_path)
#     img = img.numpy() 
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # dim = (500, 300) #set dim to whatever here
    img = tf.image.resize(img, dim)

    clt = KMeans(n_clusters=clusters)
    k_cluster = clt.fit(tf.reshape(img, [-1, 3]))
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = np.zeros(clusters)
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    # perc = dict(sorted(perc.items()))
    
    #for logging purposes
    if verbose:
        print(perc)
        print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
    #visualize results:
    if verbose:
        show_img_compar(img, palette)
        
    return perc, k_cluster.cluster_centers_


def calc_color_distance(image,camo_gen):
  '''Take the colors in generated camo and calculate their distance 
  from the primary colors in the landscape
  image :: landscape image 
  camo_gen :: camo from generator
  returns :: scalar quantity representing total distance 
  '''

  #generate the colors and proportions for landscape & camo
  prop_i, colors_i = palette_perc(image, 5, verbose=False)
  prop_c, colors_c = palette_perc(camo_gen, 5, verbose=False)

  #sort the colors by represented proportion in image/camo
  sorted_idxs_i = np.argsort(prop_i)
  sort_ci = colors_i[sorted_idxs_i]
  
  sorted_idxs_c = np.argsort(prop_c)
  sort_cc = colors_c[sorted_idxs_c]
#   sort_ci = [x for _, x in sorted(zip(prop_i, colors_i))] # this fails when there are two elements of equal value in prop_i
#   sort_cc = [x for _, x in sorted(zip(prop_c, colors_c))]

  #get minimum distance between all colors based on proportions 
  cam_list = []
  for i,color in enumerate(sort_ci):
    distance = np.linalg.norm(color - sort_cc[i]) 
    cam_list.append(distance) 
    
  return np.mean(cam_list)

