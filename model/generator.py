import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models import pix2pix

OUTPUT_CHANNELS = 1

def downstep(filters, size, stride, padding, dilation, apply_batchnorm=True, c_num=1) :
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  n = 0

  while n<c_num :
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=(stride,stride), padding=padding, dilation_rate=dilation,
                               kernel_initializer=initializer, use_bias=True))
    result.add(tf.keras.layers.ReLU(True))
    n += 1

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  return result

def upstep(filters, size, stride, padding, apply_dropout=False, c_num=1, leaky=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  up = tf.keras.Sequential()
  short = tf.keras.Sequential()
  result = tf.keras.Sequential()

  up.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=(stride,stride), padding=padding,
                                    kernel_initializer=initializer,
                                    use_bias=True))
  short.add(
    tf.keras.layers.Conv2D(filters, size-1, stride-1, padding, use_bias=True)
  )

  n = 0
  while n < c_num:
    result.add(tf.keras.layers.ReLU(True))
    result.add(
      tf.keras.layers.Conv2D(filters, size-1, stride-1, padding,
                             kernel_initializer=initializer,
                             use_bias=True)
    )
    n += 1
  if leaky :
    result.add(tf.keras.layers.LeakyReLU())

  else :
    result.add(tf.keras.layers.ReLU(True))
    result.add(tf.keras.layers.BatchNormalization())

  # if apply_dropout:
  #     result.add(tf.keras.layers.Dropout(0.5))
  return up, short, result

def Generator():
  ## 원래는 51, 71이나 그림을 생각하면 52, 72로 바꾸어 2번 줄였을 때도 13, 18로 정수로 떨어지게 바꾸었다.
  inputs = tf.keras.layers.Input(shape=(48,72,4))

  down_stack = [
    downstep(64, 3, 1, 'same', 1, apply_batchnorm=True, c_num=2), # (bs, 128, 128, 64)
    downstep(128, 3, 1, 'same', 1, apply_batchnorm=True, c_num=2), # (bs, 64, 64, 128)
    downstep(256, 3, 1, 'same', 1, apply_batchnorm=True, c_num=3), # (bs, 32, 32, 256)
  ]
  dilation = [
    downstep(512, 3, 1, 'same', 1, apply_batchnorm=True, c_num=3),  # (bs, 16, 16, 512)
    downstep(512, 3, 1, 'same', 2, apply_batchnorm=True, c_num=3),  # (bs, 8, 8, 512)
    downstep(512, 3, 1, 'same', 2, apply_batchnorm=True, c_num=3),  # (bs, 4, 4, 512)
    downstep(512, 3, 1, 'same', 1, apply_batchnorm=True, c_num=3),  # (bs, 2, 2, 512)
  ]
  up_stack = [
    upstep(256, 4, 2, 'same', apply_dropout=True, c_num=2, leaky=False), # (bs, 2, 2, 1024)
    upstep(128, 4, 2, 'same', apply_dropout=True, c_num=1, leaky=False), # (bs, 4, 4, 1024)
    upstep(128, 4, 2, 'same', apply_dropout=True, c_num=1, leaky=True), # (bs, 8, 8, 1024)
  ]
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.Sequential()
  last.add(tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1,
                                         strides=1,
                                         padding='valid',
                                         kernel_initializer=initializer,
                                         use_bias=True))# (bs, 256, 256, 3))
  last.add(tf.keras.layers.Activation('tanh'))

  x = inputs
  # Downsampling through the model
  skips = []

  for down in down_stack:
    x = down(x)
    skips.append(x)
    x = x[:,::2,::2,:]

  skips = reversed(skips)

  # Dilation through the model

  for dil in dilation:
    x = dil(x)

  # Upsampling and establishing the skip connections

  for upst, skip in zip(up_stack, skips):
    up, short, res = upst

    x = up(x)

    y = short(skip)
    x = tf.keras.layers.Concatenate()([x, y])
    x = res(x)

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# generator = Generator()
# tf.keras.utils.plot_model(generator, to_file='generator.png',show_shapes=True, dpi=64)
