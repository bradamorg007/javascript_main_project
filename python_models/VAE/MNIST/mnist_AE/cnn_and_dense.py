from keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, Input, Flatten, Reshape
from keras.models import Model
from keras import backend as KB
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import numpy as np

# Here I am going to experiment with the mnist data set for autoencoders
# mnist data set is simular to the kinds of images i will be using in my model

# import data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normaliose pixel values of images to be from 0-1

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = np.reshape(x_train, (len(x_train),28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

a = 1




# define model

# INPUT LAYER

input_layer = Input(shape=(28, 28, 1))

# ENCODER

e = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
e = MaxPooling2D(pool_size=(2, 2), padding='same')(e)
e = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D(pool_size=(2, 2), padding='same')(e)
e = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D(pool_size=(2, 2), padding='same')(e)

# maxpooling normal default is size 2x2 with stride 2 will produce a output that is half the size of the input
# here the latent space will be 4x4x8 = 128 dims = 28/2=14/2=7/2=3.5=round=4
# flatten further with dense layers for vector representation

# DENSE COMPRESSION

flat = Flatten()(e)
e = Dense(units=64, activation='relu')(flat)


latent_space = Dense(units=2, activation='relu')(e)


# DECODE

e = Dense(units=64, activation='relu')(latent_space)
e = Dense(units=128, activation='relu')(e)
e = Reshape((4, 4, 8))(e)
e = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(e)
e = UpSampling2D(size=(2,2))(e)
e = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(e)
e = UpSampling2D(size=(2,2))(e)
e = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='valid')(e)
e = UpSampling2D(size=(2,2))(e)

# need to use sigmoid to create 1 final feature map with values from 0-1 that will be the created img
decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(e)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

autoencoder.fit(x=x_train, y=x_train, epochs=10, batch_size=128, shuffle=True, verbose=2,
                validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='..\mnist_AE')])

# Plot original and reconstructions
encoder = Model(input_layer, latent_space)
pred = encoder.predict(x_test, batch_size=16)

plt.figure(figsize=(10, 10))

plt.scatter(pred[:, 0], pred[:, 1], c=y_test)
plt.colorbar()
plt.show()


decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20,4))

for i in range(n):

    # disp original

    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()

    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()

plt.show()

