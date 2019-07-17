
import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from autoencoders.autoencoder_template import AutoEncoder
import os



class VaritationalAutoEncoderSymmetric(AutoEncoder):

    def __init__(self, img_shape, latent_dimensions, batch_size):
        super().__init__(img_shape, latent_space_dims=latent_dimensions, batch_size=batch_size)


    def define_model(self):

        img_shape = self.img_shape
        latent_space_dims = self.latent_space_dims

        input_img = layers.Input(shape=img_shape)

        # ENCODER ==================================================================================================
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

        shape_before_flattening = K.int_shape(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=250, activation='relu')(x)

        z_mean = layers.Dense(units=latent_space_dims, name='z_mean')(x)
        z_log_var = layers.Dense(units=latent_space_dims, name='z_log_var')(x)

        encoder = Model(input_img, [z_mean, z_log_var], name='Encoder')
        encoder.summary()
        self.encoder = encoder


        # SAMPLER =========================================================================================
        def sample_from_latent_space(args):

            z_mean, z_log_var = args

            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dims),
                                      mean=0., stddev=1)
            latent_vector = z_mean + K.exp(z_log_var) * epsilon
            return latent_vector


        latent_space = layers.Lambda(sample_from_latent_space, name='latent_space')([encoder.output[0], encoder.output[1]])
        self.latent_space = latent_space

        # DECODER =========================================================================================
        decoder_inputs = layers.Input(shape=K.int_shape(latent_space)[1:])

        d = layers.Dense(units=250, activation='relu')(decoder_inputs)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(d)

        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)
        d = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu', strides=(2, 2))(d)

        # OUTPUT IMAGE:
        decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_space)
        self.decoder = decoder
        self.define_flag = True


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


        model = Model(input_img, z_decoded)
        model.compile(optimizer='rmsprop', loss=vae_loss)
        model.summary()
        self.model = model
        self.encoder.compile(optimizer='rmsprop', loss=vae_loss)
        self.decoder.compile(optimizer='rmsprop', loss=vae_loss)


    def load_weights(self, full_path):
        self.define_model()

        self.model.load_weights(os.path.join(full_path, 'weights_model.h5'))
        self.encoder.load_weights(os.path.join(full_path, 'weights_encoder_model.h5'))
        self.decoder.load_weights(os.path.join(full_path, 'weights_decoder_model.h5'))

        print('LOAD WEIGHTS COMPLETE')


if __name__ == "__main__":


    VAE = VaritationalAutoEncoderSymmetric(img_shape=(40, 40, 1), latent_dimensions=15, batch_size=128)
    VAE.data_prep(directory_path='../AE_data/data_seen_unseen_dynamic/', skip_files=['.json'], data_index=0, label_index=1,
                  normalize=True, remove_blanks=True, data_type='train')

    VAE.filter(keep_labels=['seen-', 'unseen-'], data_type='train')

    VAE.map_labels_to_codes()
    VAE.show_label_codes()
    VAE.define_model()
    VAE.train(epochs=50)
    VAE.inspect_model()

   # VAE.save(name='VAE', save_type='weights')