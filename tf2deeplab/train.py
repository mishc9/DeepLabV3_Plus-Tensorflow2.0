from glob import glob

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tf2deeplab.deeplab import DeepLabV3Plus
from tf2deeplab.utils import load_data

print('TensorFlow', tf.__version__)

batch_size = 24
H, W = 512, 512
num_classes = 34

image_list = sorted(glob(
    'cityscapes/dataset/train_images/*'))
mask_list = sorted(glob(
    'cityscapes/dataset/train_masks/*'))

val_image_list = sorted(glob(
    'cityscapes/dataset/val_images/*'))
val_mask_list = sorted(glob(
    'cityscapes/dataset/val_masks/*'))

print('Found', len(image_list), 'training images')
print('Found', len(val_image_list), 'validation images')

for i in range(len(image_list)):
    assert image_list[i].split(
        '/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]

for i in range(len(val_image_list)):
    assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[
               0] == val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]

train_dataset = tf.data.Dataset.from_tensor_slices((image_list,
                                                    mask_list))
train_dataset = train_dataset.shuffle(buffer_size=128)
train_dataset = train_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(train_dataset)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                  val_mask_list))
val_dataset = val_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepLabV3Plus(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)
callbacks = [mc, tb]

model.fit(train_dataset,
          steps_per_epoch=len(image_list) // batch_size,
          epochs=300,
          validation_data=val_dataset,
          validation_steps=len(val_image_list) // batch_size,
          callbacks=callbacks)
