import numpy as np
import pickle
import os
import time
import datetime
import keras.layers
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
import model.generator as gen
import model.samplegen as sgen
import model.discriminator as disc
from model.loss import discriminator_loss, generator_loss
from model.optimizer import discriminator_optimizer, generator_optimizer
from util.plotcuv import image
import data.get_data as dataset
# train_in, train_real, C_max, C_min, W_max, W_min = dataset.get_train_data()
# test_in = dataset.get_test_data(C_max, C_min, W_max, W_min)

with open('train_input.pickle', 'rb') as f:
  train_in = pickle.load(f)
  train_in = train_in.astype(np.float32)
  C_max = np.max(train_in[:,:,:,0])

with open('train_real.pickle', 'rb') as f:
  train_real = pickle.load(f)
  train_real = train_real.astype(np.float32)

with open('test_input.pickle', 'rb') as f:
  test_in = pickle.load(f)
  test_in = test_in.astype(np.float32)

print(train_in.shape, train_real.shape, test_in.shape)

generator = gen.Generator()
discriminator = disc.Discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))
  prediction_att = test_input[:,:,:,1:-1]
  prediction = tf.concat((prediction,prediction_att),axis=3)
  print(tar[0].shape)
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    # getting the pixel values between [0, 1] to plot it.
    image(title[i],display_list[i],C_max)
  # for i in range(3):
  #   plt.subplot(1, 3, i + 1)
  #   plt.title(title[i])
  #   # getting the pixel values between [0, 1] to plot it.
  #   plt.imshow(display_list[i])
  #   plt.axis('off')
  # plt.show()

EPOCHS = 150

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)
    disc_real_output = discriminator([input_image, target], training=True)
    gen_output_att = input_image[:,:,:,1:-1]
    gen_output = tf.concat([gen_output,gen_output_att],axis=3)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_in, train_real, epochs,test_in):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_in), tf.constant(train_real)))
    train_ds = train_ds.batch(batch_size=1)
    for example_input, example_target in train_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if n==5000 :
        break
      if (n + 1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

fit(train_in, train_real, EPOCHS, test_in)

for inp, tar in test_in.take(50):
  generate_images(generator, inp, tar)