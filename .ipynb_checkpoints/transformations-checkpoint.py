import tensorflow as tf
import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image


def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image


def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


def preprocess_image_test(image, label):
  image = normalize(image)
  return image


def cutout_and_replace(orig_landscape, camo, cutout_size=16):
    H, W = orig_landscape.shape[1:3]
    assert cutout_size <= H//2

    # Determining square location
    H_start = int(H//2)
    x_start = np.random.randint(0, W - cutout_size)
    y_start = np.random.randint(H_start, H - cutout_size)

    # Square indices
    xs = list(range(x_start, x_start+cutout_size))
    ys = list(range(y_start, y_start+cutout_size))
    list_idxs = np.dstack(np.meshgrid(ys, xs)).reshape(-1, 2)

    # Resizing camo patch
    resized_camo_patch = tf.image.resize(camo, [cutout_size, cutout_size])
    resized_camo_patch = tf.transpose(resized_camo_patch, perm=[0, 2, 1, 3]) # Transpose image
    list_camo_pix = tf.reshape(resized_camo_patch, [cutout_size**2, 3])

    # Replacing with resized camo
    updated_landscape = tf.tensor_scatter_nd_update(orig_landscape[0], list_idxs, list_camo_pix)

    return tf.expand_dims(updated_landscape, 0) # Convert back to (1, H, W, 3)

