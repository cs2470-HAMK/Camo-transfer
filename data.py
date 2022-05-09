import tensorflow as tf
import numpy as np
import pandas as pd

import os

import glob
from PIL import Image

def data_preprocessor(file_path, limit=None):

    #create dataframe to store image paths and images:

    images_dict = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob.glob(os.path.join(file_path, '', '*.jpg'))}

    df = pd.DataFrame(images_dict.items())
    #rename columns for clarity
    df = df.rename(columns={df.columns[0]: "image_id"})
    df = df.rename(columns={df.columns[1]: "image_path"})
    if limit:
      df = df[:limit]
    df["image"] = df['image_path'].map(lambda x: np.asarray(Image.open(x).resize((256, 256))))

    imgs = df["image"]
    inputs = np.array(imgs)/(255.0) # normalization

    interim_inputs = [j for i, j in enumerate(inputs)]
    inp_reshape = tf.reshape(interim_inputs, (-1, 256, 256, 3))
    train_imgs = np.asarray(inp_reshape, dtype= np.float32)

    # train_imgs = tf.random.shuffle(final_inputs)

    og_images = []
    for tensor in train_imgs:
        og_images.append(tensor)

    # Apply data augmentation to populate some data
    # With data augmentation to prevent overfitting
    """
    lst_saturated = []
    for i in range(len(train_imgs)):
        saturation_played_1_3 = tf.image.adjust_saturation(train_imgs[i], 1.3)
        saturation_played_1_6 = tf.image.adjust_saturation(train_imgs[i], 1.6)
        saturation_played_1_9 = tf.image.adjust_saturation(train_imgs[i], 1.9)
        lst_saturated.append(saturation_played_1_3)
        lst_saturated.append(saturation_played_1_6)
        lst_saturated.append(saturation_played_1_9)

    res_list = [y for x in [og_images, lst_saturated] for y in x]
    """

    tensor_converted_images = tf.convert_to_tensor(og_images)
    image_dataset = tf.data.Dataset.from_tensor_slices(tensor_converted_images)

    ds_size = tf.data.experimental.cardinality(image_dataset)
    train_split=0.8
    test_split=0.2
    shuffle_size=296

    Shuffle=True
    if Shuffle:
    # Specify seed to always have the same split distribution between runs
        ds = image_dataset.shuffle(shuffle_size, seed=12)

    train_size = int(np.ceil(train_split * int(ds_size)))
    test_size = int(np.ceil(test_split * int(ds_size)))

    train_ds = ds.take(train_size)   
    test_ds = ds.take(test_size)

    train_size_lst = []
    for img in train_ds:
        train_size_lst.append(img)

    train_imgs_arrays = []
    for tensor_images in train_size_lst:
        array_img = np.asarray(tensor_images)
        train_imgs_arrays.append(array_img)

    # dark_skin_sample = train_imgs_arrays[20]

    return train_ds, test_ds # , dark_skin_sample