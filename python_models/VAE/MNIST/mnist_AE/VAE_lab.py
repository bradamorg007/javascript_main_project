from keras import layers
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model


def vae_loss(original_input, decoded_output):
    # add distangled = VAE introduce the B parameter to scale the KL loss my multiplication
    original_input = K.flatten(original_input)
    decoded_output = K.flatten(decoded_output)

    z_mean1, z_log_var1 = encoder.output[0], encoder.output[1]

    # define normal autoencoder loss
    AE_loss = keras.metrics.binary_crossentropy(original_input, decoded_output)

    # Kl divergence for gaussian fitting
    KL_loss = -5e-4 * K.mean(1 + z_log_var1 - K.square(z_mean1) - K.exp(z_log_var1), axis=-1)

    # combine and return mean of losses
    return K.mean(AE_loss + KL_loss)


def data_prep(keep_labels):
    # DATA PREP=============================================================================================================
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    keep_index2 = []
    for i, label in enumerate(y_test):

        for item in keep_labels:

            if label == item:
                keep_index2.append(i)
                break

    x_test = x_test[keep_index2]
    y_test = y_test[keep_index2]

    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_test, y_test

    # Current VAE has been trained to recognise 0 & 1s now will get model to predict a unknown class

latent_space_dims = 3

VAE = load_model('..\mnist_AE\models\VAE_2dig_01.h5',
                 custom_objects={'latent_space_dims': latent_space_dims}, compile=False)

encoder = load_model('..\mnist_AE\models\VAE_encoder_2dig_01.h5',
                     custom_objects={'latent_space_dims': latent_space_dims}, compile=False)

keep_labels = [7]
x_test_unseen, y_test_unseen = data_prep(keep_labels)

keep_labels = [0, 1]
x_test_seen, y_test_seen = data_prep(keep_labels)

# PLOTTING & METRICS===================================================================================================

# plot the latent space of the VAE
# encoder = Model(input_img, [z_mean, z_log_var, latent_space], name='encoder')

z_mean_unseen, _, = encoder.predict(x_test_unseen, batch_size=16)
z_mean_seen, _, = encoder.predict(x_test_seen, batch_size=16)

fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
p = ax.scatter(z_mean_seen[:, 0], z_mean_seen[:, 1], z_mean_seen[:, 2], c=y_test_seen)
d = ax.scatter(z_mean_unseen[:, 0], z_mean_unseen[:, 1], z_mean_unseen[:, 2], color='k')
fig.colorbar(p, fraction=0.089, )
fig.show()

encoder.save('..\mnist_AE\models\VAE_encoder_2dig_01.h5')

# Plot comparisons between original and decoded images with test data
decoded_imgs = VAE.predict(x_test_unseen)
n = 10
plt.figure(figsize=(20, 4))
plt.title('Original vs Reconstructed Images: Unseen Data')

for i in range(n):
    # disp original

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_unseen[i].reshape(28, 28))
    plt.gray()

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()

plt.show()

decoded_imgs = VAE.predict(x_test_seen)
n = 10
plt.figure(figsize=(20, 4))
plt.title('Original vs Reconstructed Images: Seen Data')

for i in range(n):
    # disp original

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_seen[i].reshape(28, 28))
    plt.gray()

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()

plt.show()

VAE.compile(optimizer='rmsprop', loss=vae_loss)

history_seen = VAE.evaluate(x_test_seen)

history_unseen = VAE.evaluate(x_test_unseen)

a = 0
