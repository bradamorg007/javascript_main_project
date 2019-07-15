
import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from test_code.autoencoders.autoencoder import AutoEncoder


class VaritationalAutoEncoder(AutoEncoder):

    def __init__(self, img_shape, latent_dimesnions, batch_size):
        super().__init__(img_shape, latent_space_dims=latent_dimesnions, batch_size=batch_size)


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

if __name__ == "__main__":

    VAE = VaritationalAutoEncoder((40, 40, 1), 2, 16)
    VAE.define_model()
    a = 1