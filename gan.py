from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from support.image_save import image_for_snapshot , image_from_array
from support.glove import Glove
from keras import backend as K
import numpy as np
from PIL import Image
import os

class Gan(object):
    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width = 16
        self.img_height = 16
        self.img_channels = 3
        self.random_input_dim = 20
        self.text_input_dim = 100
        self.config = None
        self.glove_source_dir_path = './glove'
        self.glove_model = Glove()

    def create_model(self):
        init_img_width = self.img_width // 4
        init_img_height = self.img_height // 4

        random_input = Input(shape=(self.random_input_dim,))
        text_input1 = Input(shape=(self.text_input_dim,))
        random_dense = Dense(self.random_input_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(128 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height, 128),
                                  input_shape=(128 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, text_input1], generator_output)

        self.generator.compile(loss='mean_squared_error', optimizer="SGD")

        # print('generator: ', self.generator.summary())

        text_input2 = Input(shape=(self.text_input_dim,))
        text_layer2 = Dense(1024)(text_input2)

        img_input2 = Input(shape=(self.img_width, self.img_height, self.img_channels))
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(
            img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)

        merged = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        self.discriminator = Model([img_input2, text_input2], discriminator_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        # print('discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, text_input1])

        self.model = Model([random_input, text_input1], model_output)
        self.discriminator.trainable = False

        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        # print('generator-discriminator: ', self.model.summary())


    def load_model(self):
        config_file_path = './weights/config.npy'
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.text_input_dim = self.config['text_input_dim']
        self.glove_source_dir_path = self.config['glove_source_dir_path']
        self.create_model()
        self.glove_model.load_glove(self.glove_source_dir_path)
        self.generator.load_weights('./weights/gen.h5')
        self.discriminator.load_weights('./weights/disc.h5')

    def fit(self, image_label_pairs, epochs, batch_size, snapshot_dir_path , snapshot_interval ,  model_dir_path):

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path

        self.glove_model.load_glove(self.glove_source_dir_path)

        config_file_path = './weights/config.npy'

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        self.create_model()

        for epoch in range(epochs):
            print("Epoch :", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator

                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_text(text)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_batch = np.array(image_batch)

                generated_images = self.generator.predict([noise, text_batch], verbose=0)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)),
                                                            np.concatenate((text_batch, text_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))

                # Step 2: train the generator
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                if epoch % 100 == 0 and (batch_index == 0 or batch_index ==1) :
                    print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))
                    print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))
                    self.save_intermediate_result(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch, batch_index=batch_index)
                    self.generator.save_weights('./weights/gen.h5', True)
                    self.discriminator.save_weights('./weights/disc.h5' , True)

        self.generator.save_weights('./weights/gen.h5', True)
        self.discriminator.save_weights('./weights/disc.h5' , True)


    def generate_image_from_text(self, text):
        noise = np.zeros(shape=(1, self.random_input_dim))
        encoded_text = np.zeros(shape=(1, self.text_input_dim))
        encoded_text[0, :] = self.glove_model.encode_text(text)
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
        generated_images = self.generator.predict([noise, encoded_text], verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_intermediate_result(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = image_for_snapshot(generated_images)
        image_from_array(image).save(
            os.path.join(snapshot_dir_path , 'result' + str(epoch) + "-" + str(batch_index) + ".jpg"))

