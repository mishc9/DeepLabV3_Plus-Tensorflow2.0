import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tf2deeplab.inference import w, h, model, id_to_color


def get_image(image_path, img_height=800, img_width=1600, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
    return img


def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask


def load_data(image_path, mask_path, H=512, W=512):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(
        mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    return image, mask


def pipeline(image, video=True, return_seg=False, fname='', folder=''):
    global b
    alpha = 0.5
    dims = image.shape
    image = cv2.resize(image, (w, h))
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)

    img_color = image.copy()
    for i in np.unique(y):
        if i in id_to_color:
            img_color[y == i] = id_to_color[i]
    disp = img_color.copy()
    if video:
        cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
        return img_color
    if return_seg:
        return img_color / 255.
    else:
        cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
#         plt.figure(figsize=(20, 10))
#         out = np.concatenate([image/255, img_color/255, disp/255], axis=1)

#         plt.imshow(img_color/255.0)
#         plt.imshow(out)
        return cv2.imwrite(f'outputs/{folder}/{fname}', cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))