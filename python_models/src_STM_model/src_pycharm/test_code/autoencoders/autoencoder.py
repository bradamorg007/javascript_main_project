from keras import layers
from keras import backend as K
import keras
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from test_code.file_extractor import FileExtractor
from nn import NeuralNetwork

class AutoEncoder:

    def __init__(self, img_shape, latent_space_dims, batch_size):

       self.img_shape = img_shape
       self.latent_space_dims = latent_space_dims
       self.encoder = None
       self.latent_space = None
       self.decoder = None
       self.model = None

       self.batch_size = batch_size
       self.x_train = None
       self.y_train = None
       self.x_test = None
       self.y_test = None
       self.blanks_detected = [0, []]
       self.history = None

       self.label_table_train = None
       self.label_keys_train = None
       self.label_table_test = None
       self.label_keys_test = None

       self.define_flag = False
       self.data_flag = False
       self.train_flag = False
       self.label_config_flag = True




    def data_prep(self, directory_path, skip_files, data_index,
                  label_index, normalize, remove_blanks, data_type):

        data = FileExtractor.extract(directory_path=directory_path, skip_files=skip_files)

        x = np.zeros(shape=(len(data), data[1][data_index].shape[0], data[1][data_index].shape[1] , 1))
        y = []

        for i in range(len(x)):
            sample = data[i][data_index]
            label = data[i][label_index]


            if remove_blanks:

                if sample.min() != sample.max():
                    if normalize:
                        sample = sample / 255

                        x[i] = np.reshape(sample.astype('float32'), newshape=sample.shape + (1,))
                        y.append(label)
                else:
                    self.blanks_detected[0] += 1
                    self.blanks_detected[1].append(i)




        y = np.array(y)

        if data_type == 'train':
            self.x_train = x
            self.y_train = y
            self.data_flag = True
            self.label_config('train')

        elif data_type == 'test':
            self.x_test = x
            self.y_test = y
            self.data_flag = True
            self.label_config('test')

        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


    def label_config(self, data_type):

        y = None
        if data_type == 'train':
            y = self.y_train

        elif data_type == 'test':
            y = self.y_test
        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


        if self.data_flag:
             label_table = AutoEncoder.count_unquie(y)

             if data_type == 'train':
                self.label_table_train = label_table
                self.label_keys_train = label_table.keys()

             elif data_type == 'test':
                 self.label_table_test = label_table
                 self.label_keys_test = label_table.keys()

        else:
            raise ValueError("ERROR Label Config: Please use the data prep function before using this function")


    def filter(self, keep_labels, data_type):

        label_table = None
        if data_type == 'train':
            label_table = self.label_table_train

        elif data_type == 'test':
            label_table = self.label_table_test
        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


        if self.label_config_flag:
            if isinstance(keep_labels, list):

                keep_labels_indexes = []
                for label in keep_labels:

                    lookup = label_table.get(label)

                    if lookup is not None:
                       keep_labels_indexes.append(lookup.get('indices'))
                    else:
                        raise ValueError('ERROR filter: Element in keep labels list does not exist in the label _table')

                if data_type == 'train':
                    self.x_train = self.x_train[keep_labels_indexes]
                    self.y_train = self.y_train[keep_labels_indexes]
                elif data_type == 'test':
                    self.x_test = self.x_test[keep_labels_indexes]
                    self.y_test = self.y_test[keep_labels_indexes]


            else:
                raise ValueError('ERROR filter: keep_labels must of type list')
        else:
            raise ValueError("ERROR filter: Please use the label config function before using this function")


    def train(self, epochs):

        if self.define_flag and self.data_flag:
            # =======================================================================================================================
            history = self.model.fit(x=self.x_train, y=self.x_train,
                                   shuffle=True, epochs=epochs, batch_size=self.batch_size,
                                   validation_split=0.1, verbose=2)

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

        z_mean, _, = self.encoder.predict(self.x_train, batch_size=self.batch_size)

        fig = plt.figure(figsize=(6, 6))
        #ax = Axes3D(fig)
        plt.scatter(x=z_mean[:, 0], y=z_mean[:, 1])
        fig.show()

        # Plot comparisons between original and decoded images with test data
        decoded_imgs = self.VAE.predict(self.x_train)
        n = 10
        plt.figure(figsize=(20, 4))

        image_samples = np.random.randint(0, len(self.x_train), size=n)

        for i in range(n):
            # disp original

            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.x_train[image_samples[i]].reshape(40, 40))
            plt.gray()

            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[image_samples[i]].reshape(40, 40))
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


    @staticmethod
    def count_unquie(input):
        table = {}

        for i, rows in enumerate(input):

            # try to retrieve a label key, if it exists append the freq count and add i to indices

            lookup = table.get(rows)

            if lookup is not None:

                lookup['freq'] += 1
                lookup['indices'].append(i)

            else:
                # if it doesnt exist in the dict then make a new entry
                table[rows] = {'freq': 1, 'indices': [i]}

        return table



if __name__ == '__main__':

    VAE = AutoEncoder(img_shape=(40, 40, 1), latent_space_dims=2, batch_size=128)
    VAE.data_prep(directory_path="../data_seen_augment/", skip_files=['.json'], data_index=0, label_index=1,
                  normalize=True, remove_blanks=True, data_type='train')


    VAE.train(epochs=3)
    VAE.inspect_model()
    #VAE.save(name='trained_digits_8', save_type='weights')
    #
    # VAE.load_weights(full_path='..\mnist_AE\models/test')
    # print('Seen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))
    #
    # VAE.data_prep(keep_labels=[7])
    # print('Unseen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))

