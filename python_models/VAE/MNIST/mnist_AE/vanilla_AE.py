from keras import layers
from keras.models import Model
from keras import backend as KB
import keras
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D


# Here I am going to experiment with the mnist data set for autoencoders
# mnist data set is simular to the kinds of images i will be using in my model


# define model

# INPUT LAYER

def train(data, epochs, batch_size, latent_space_dims):

    x_train, y_train, x_test, y_test = data


    input_layer = layers.Input(shape=(28, 28, 1))

    # ENCODER

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # record the dimension size before flattening for parameter prediction
    shape_before_flattening = KB.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=32, activation='relu')(x)


    # DENSE COMPRESSION
    latent_space = layers.Dense(units=latent_space_dims, name='latent_space')(x)


    # DECODE

    # upsample sense vector. prod = shape_bef = 2x2x64 e.g so = 256 long flat vector
    x = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(latent_space)
    x = layers.Reshape(target_shape=shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu', strides=(2, 2))(x)
    decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                activation='sigmoid', name='decoded_img')(x)


    autoencoder = Model(input_layer, decoded_img)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(autoencoder.summary())

    #  callbacks=[TensorBoard(log_dir='..\mnist_AE')]
    history = autoencoder.fit(x=x_train, y=x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2,
                    validation_data=(x_test, x_test))

    encoder = Model(input_layer, latent_space)

    return autoencoder, encoder, history



def save(name, autoencoder, encoder):
    folder = "..\mnist_AE\models"

    if os.path.exists(os.path.join(folder, name)) == False:
        os.mkdir(os.path.join(folder, name))

    autoencoder.save(os.path.join(folder,name, 'vanilla_model.h5'))
    encoder.save(os.path.join(folder, name, 'vanilla_encoder_model.h5'))


def load(path):

    autoencoder = keras.models.load_model(os.path.join(path, 'vanilla_model.h5'))
    encoder = keras.models.load_model(os.path.join(path, 'vanilla_encoder_model.h5'))

    return autoencoder, encoder


def data_prep(keep_labels):

    # DATA PREP=============================================================================================================
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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

    return [x_train, y_train, x_test, y_test]



def inspect_model(autoencoder, encoder, data):

    # Plot original and reconstructions
    x_train, y_train, x_test, y_test = data
    pred = encoder.predict(x_test, batch_size=16)

    if len(pred[0]) == 2:
        plt.figure(figsize=(10, 10))

        plt.scatter(pred[:, 0], pred[:, 1], c=y_test)
        plt.colorbar()
        plt.show()

    elif len(pred[0]) == 3:

        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        p = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=y_test)
        fig.colorbar(p)
        fig.show()

    else:
        print('WARNING: MODEL LATENT SPACE EXCEEDS INTERPRETABLE (ls_dims > 3D) PLOTTING DIMENSIONS ')


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


def scatter_plot_comparison(pred_seen, pred_unseen, y_test_seen):


    if len(pred_seen[0]) == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        p = ax.scatter(pred_seen[:, 0], pred_seen[:, 1], pred_seen[:, 2], c=y_test_seen)
        d = ax.scatter(pred_unseen[:, 0], pred_unseen[:, 1], pred_unseen[:, 2], color='k')
        fig.colorbar(p, fraction=0.089, )
        fig.show()
    elif len(pred_seen[0]) == 2:
        fig = plt.figure(figsize=(6, 6))
        p = plt.scatter(pred_seen[:, 0], pred_seen[:, 1], c=y_test_seen)
        d = plt.scatter(pred_unseen[:, 0], pred_unseen[:, 1], color='k')
        fig.colorbar(p, fraction=0.089, )
        fig.show()


def lab_model():
    autoencoder, encoder = load('..\mnist_AE\models/trained_vanilla')
    x_train_unseen, y_train_unseen, x_test_unseen, y_test_unseen = data_prep(keep_labels=[2])

    x_train_seen, y_train_seen, x_test_seen, y_test_seen = data_prep(keep_labels=[0, 1])



    pred_unseen = encoder.predict(x_test_unseen, batch_size=16)
    pred_seen = encoder.predict(x_test_seen, batch_size=16)
    scatter_plot_comparison(pred_seen, pred_unseen, y_test_seen=y_test_seen)

    # Plot comparisons between original and decoded images with test data
    decoded_imgs = autoencoder.predict(x_test_unseen)
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

    decoded_imgs = autoencoder.predict(x_test_seen)
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


    seen_RE = autoencoder.evaluate(x_test_seen, x_test_seen, batch_size=16)

    unseen_RE =  autoencoder.evaluate(x_test_unseen, x_test_unseen, batch_size=16)


    plt.figure(figsize=(6, 6))
    plt.bar(x=np.arange(len([seen_RE, unseen_RE])), height=[seen_RE, unseen_RE], tick_label=['seen_RE', 'unseen_RE'])
    plt.show()


if __name__ == '__main__':

    # data = data_prep(keep_labels=[0,1])
    # autoencoder, encoder, history = train(data, epochs=50, batch_size=16, latent_space_dims=2)
    # inspect_model(autoencoder, encoder, data)
    # save(name='trained_vanilla', autoencoder=autoencoder, encoder=encoder)

    lab_model()

