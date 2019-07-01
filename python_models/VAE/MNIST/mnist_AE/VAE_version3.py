from keras import layers
from keras import backend as K
import keras
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class VariationalAutoEncoder:

    def __init__(self, img_shape=(28, 28, 1), latent_space_dims=3, batch_size=16):

       self.img_shape = img_shape
       self.latent_space_dims = latent_space_dims
       self.encoder = None
       self.latent_space = None
       self.decoder = None
       self.VAE = None
       self.batch_size = batch_size
       self.x_train = None
       self.y_train = None
       self.x_test = None
       self.y_test = None
       self.history = None

       self.define_flag = False
       self.data_flag = False
       self.train_flag = False


    def define_model(self):

        # img_shape = (28, 28, 1)
        # latent_space_dims = 3

        img_shape = self.img_shape
        latent_space_dims = self.latent_space_dims

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

        encoder = Model(input_img, [z_mean, z_log_var], name='Encoder')
        encoder.summary()
        self.encoder = encoder
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


        latent_space = layers.Lambda(sample_from_latent_space, name='latent_space')([encoder.output[0], encoder.output[1]])
        self.latent_space = latent_space
        # ==================================================================================================

        # DECODER =========================================================================================
        decoder_inputs = layers.Input(shape=K.int_shape(latent_space)[1:])

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

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_space)
        self.decoder = decoder
        self.define_flag = True


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


        VAE = Model(input_img, z_decoded)
        VAE.compile(optimizer='rmsprop', loss=vae_loss)
        VAE.summary()
        self.VAE = VAE
        self.encoder.compile(optimizer='rmsprop', loss=vae_loss)
        self.decoder.compile(optimizer='rmsprop', loss=vae_loss)


    def data_prep(self, keep_labels):

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

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data_flag = True


    def train(self, epochs):

        if self.define_flag and self.data_flag:
            # =======================================================================================================================
            history = self.VAE.fit(x=self.x_train, y=self.x_train,
                                   shuffle=True, epochs=epochs, batch_size=self.batch_size,
                                   validation_data=(self.x_test, self.x_test), verbose=2)

            self.history = history
            self.train_flag = True

        else:
            raise ValueError('ERROR: THE MODEL AND THE DATA MUST BE DEFINED BEFORE TRAIN CAN BE CALLED')


    def save(self, name, save_type):

        folder = "..\mnist_AE\models"

        if os.path.exists(os.path.join(folder, name)) == False:
            os.mkdir(os.path.join(folder, name))


        if save_type == 'model':

            self.VAE.save(os.path.join(folder, name, 'VAE_full_model.h5'))
            self.encoder.save(os.path.join(folder, name, 'encoder_model.h5'))
            self.latent_space.save(os.path.join(folder, name, 'latent_space_model.h5'))
            self.decoder.save(os.path.join(folder, name, 'decoder_model.h5'))

            self.VAE.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE MODEL COMPLETE')

        elif save_type == 'weights':
            self.VAE.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE WEIGHTS COMPLETE')


    def load_weights(self, full_path):

        self.define_model()

        self.VAE.load_weights(os.path.join(full_path, 'weights_model.h5'))
        self.encoder.load_weights(os.path.join(full_path, 'weights_encoder_model.h5'))
        self.decoder.load_weights(os.path.join(full_path, 'weights_decoder_model.h5'))

        print('LOAD WEIGHTS COMPLETE')


    def inspect_model(self):
        # PLOTTING & METRICS===================================================================================================

        # plot the latent space of the VAE
        #encoder = Model(input_img, [z_mean, z_log_var, latent_space], name='encoder')

        z_mean, _, = self.encoder.predict(self.x_test, batch_size=self.batch_size)

        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        p = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=self.y_test)
        fig.colorbar(p)
        fig.show()

        # Plot comparisons between original and decoded images with test data
        decoded_imgs = self.VAE.predict(self.x_test)
        n = 10
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # disp original

            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.x_test[i].reshape(28, 28))
            plt.gray()

            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()

        plt.show()

        # Plot Traning and validation reconstruction error
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(loss) + 1)

        plt.figure()

        plt.plot(epochs, loss, 'g', label='training loss')
        plt.plot(epochs, val_loss, 'r', label='validation loss')
        plt.title('Training and Validation Reconstruction Error')
        plt.legend()

        plt.show()



if __name__ == '__main__':

    VAE = VariationalAutoEncoder(img_shape=(28, 28, 1), latent_space_dims=3, batch_size=16)
    VAE.data_prep(keep_labels=[8])
    VAE.define_model()
    VAE.train(epochs=10)
    VAE.inspect_model()
    VAE.save(name='trained_digits_8', save_type='weights')
    #
    # VAE.load_weights(full_path='..\mnist_AE\models/test')
    # print('Seen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))
    #
    # VAE.data_prep(keep_labels=[7])
    # print('Unseen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))


