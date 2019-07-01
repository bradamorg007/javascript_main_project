from keras import layers
from keras import backend as K
import keras
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img_shape = (28, 28, 1)
latent_space_dims = 3

input_img = layers.Input(shape=img_shape)

# ENCODER ==================================================================================================
# Use convolution layers to feature extract and downsample image. use stride=2 for downsampling rather than maxpool
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

# record the dimension size before flattening for parameter prediction
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(units=32, activation='relu')(x)

# Just use linear activation here just the sum and dot products
z_mean = layers.Dense(units=latent_space_dims, name='z_mean')(x)
z_log_var = layers.Dense(units=latent_space_dims, name='z_log_var')(x)


# ==================================================================================================

# SAMPLER =========================================================================================
def sample_from_latent_space(args):
    # We are restricting our latent space to fit to a normal distribution thus we can sample an encoding vector
    # from that distribution using the encoded parameters

    z_mean, z_log_var = args

    # define random tensor of small values adds stochastisity
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dims),
                              mean=0., stddev=1)
    # now sample from the distribution
    latent_vector = z_mean + K.exp(z_log_var) * epsilon
    return latent_vector


latent_vector = layers.Lambda(sample_from_latent_space, name='latent_vectors')([z_mean, z_log_var])
# ==================================================================================================

# DECODER =========================================================================================
decoder_inputs = layers.Input(shape=K.int_shape(latent_vector)[1:])

# upsample sense vector. prod = shape_bef = 2x2x64 e.g so = 256 long flat vector
d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(decoder_inputs)

# reshape for cnn processing
d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)

# now reverse normal conv2d to upscale the sampled latent vector back to an image size of the original input

d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu', strides=(2, 2))(d)

# finally produce one final feature map of size original image and use sigmoid to produce 0-1 pixel values
# for the decoded image

decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                            activation='sigmoid', name='decoded_img')(d)

decoder = Model(decoder_inputs, decoded_img)
z_decoded = decoder(latent_vector)


# Define Custom loss function
# Use binary cross entropy to to assess how different decoded img is from orginal, model will seperate classes geometrically]
# use KL divergence which assess how similar two distributions are. in this case we want to assess the simularity between
# the latent space distribution and normal gaussian. we want the differences thus error to be minimum so our latent space
# resembles or fits to something closely simular to a gassian. allowing for highly structured and interpretable latent_spaces
# we add both losses together. binary_cross makes sure the model separates classes into geometrically seperable
# distributions whilst KL makes sure all distributions lay close to the centre as per the shape of gaussian
# (mean = 0, stddev=1) this makes the space continuous for interpolation but also allows us to know where abouts in
# the latent space we should sample from, which is near the centre that way we know we will get a meaningful decoded
# image back. normal autoencoders can place the class distributions anywhere thus its highly likely most places we sample
# from wont be within any of the class distributions and will just be meaningless random noise.

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, original_input, decoded_output):
        # add distangled = VAE introduce the B parameter to scale the KL loss my multiplication
        original_input = K.flatten(original_input)
        decoded_output = K.flatten(decoded_output)

        # define normal autoencoder loss
        AE_loss = keras.metrics.binary_crossentropy(original_input, decoded_output)

        # Kl divergence for gaussian fitting
        KL_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        # combine and return mean of losses
        return K.mean(AE_loss + KL_loss)

    def call(self, inputs):
        original_input = inputs[0]
        decoded_output = inputs[1]

        loss = self.vae_loss(original_input, decoded_output)
        self.add_loss(loss, inputs=inputs)

        return original_input


y = CustomVariationalLayer()([input_img, z_decoded])

VAE = Model(input_img, y)
VAE.compile(optimizer='rmsprop', loss=None)
VAE.summary()

# DATA PREP=============================================================================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

(old_x_train, old_y_train), (old_x_test, old_y_test) = (x_train, y_train), (x_test, y_test)

keep_labels = [0, 1]

keep_index1 = []
for i, label in enumerate(y_train):

    for item in keep_labels:

        if label == item:
            keep_index1.append(i)
            break

keep_index2 = []
for i, label in enumerate(y_test):

    for item in keep_labels:

        if label == item:
            keep_index2.append(i)
            break

x_train = x_train[keep_index1]
y_train = y_train[keep_index1]
x_test = x_test[keep_index2]
y_test = y_test[keep_index2]

x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape + (1,))
# =======================================================================================================================


batch_size = 16
history = VAE.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None),
                  verbose=2)

VAE.save('..\mnist_AE\models\VAE_2dig_01.h5')

# PLOTTING & METRICS===================================================================================================

# plot the latent space of the VAE
encoder = Model(input_img, [z_mean, z_log_var, latent_vector], name='encoder')

z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
p = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test)
fig.colorbar(p)
fig.show()

encoder.save('..\mnist_AE\models\VAE_encoder_2dig_01.h5')


# Plot comparisons between original and decoded images with test data
decoded_imgs = VAE.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # disp original

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()

plt.show()


# Plot Traning and validation reconstruction error
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'g', label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('Training and Validation Reconstruction Error')
plt.legend()

plt.show()
