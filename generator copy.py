from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Multiply, Add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from tensorflow import keras
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal


def resisudal_block(filters, size):
    result = tf.keras.Sequential()
    result.add(downsample(filters, size, strides=1, activation='ReLU'))
    result.add(downsample(filters, size, strides=1, activation='None'))

    return result


def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02, mean=0.0)
    g = Conv2D(n_filters, (3, 3), padding='same',
               kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Concatenate()([g, input_layer])
    return g


k_init = tf.keras.initializers.random_normal(stddev=0.008, seed=101)
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()


def build_generator(n_resnet=9):
    # height, width of input image changed because of error in output
    inputs = tf.keras.Input(shape=[256, 256, 3])

    init = RandomNormal(stddev=0.02, mean=0.0)

    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(inputs)
    g = InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)

    g = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)

    g = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)

    g = Conv2D(512, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)

    # g = Conv2D(512, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    # g = InstanceNormalization(axis=-1)(g)
    # g = tf.keras.layers.Activation('relu')(g)

    for _ in range(n_resnet):
        g = resnet_block(256, g)
    g = Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    conv4_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init, activation='relu',
                                     bias_initializer=b_init, kernel_regularizer=regularizer)(g)
    conv4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init, activation='relu',
                                     bias_initializer=b_init, kernel_regularizer=regularizer)(conv4_1)
    conv4_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init, activation='relu',
                                     bias_initializer=b_init, kernel_regularizer=regularizer)(conv4_2)
    conv4_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init, activation='relu',
                                     bias_initializer=b_init, kernel_regularizer=regularizer)(conv4_3)
    conv4_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal(seed=101),
                                     bias_initializer=b_init, kernel_regularizer=regularizer)(conv4_4)
    conc4 = tf.add(conv4_5, conv4_1)
    conv4 = tf.keras.activations.relu(conc4)

    ##### Decoding Layers #####
    deconv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal(seed=101),
                                             kernel_regularizer=regularizer)(conv4)
    deconv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal(seed=101),
                                             kernel_regularizer=regularizer)(deconv)

    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init, activation='relu',
                                  bias_initializer=b_init, kernel_regularizer=regularizer)(deconv)
    conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal(seed=101),
                                  bias_initializer=b_init, kernel_regularizer=regularizer)(conv)
    conc = tf.add(conv, inputs)

    outputs = tf.keras.layers.experimental.preprocessing.Resizing(
        256, 256, interpolation='bilinear')(conc)

    return Model(inputs=inputs, outputs=outputs)


def downsample(filters, size, apply_batchnorm=True, strides=2, padding='same', activation='LeakyReLU'):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding,
                               use_bias=True))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if activation == 'ReLU':
        result.add(tf.keras.layers.ReLU())
    elif activation == 'LeakyReLU':
        result.add(tf.keras.layers.LeakyReLU())

    return result


def build_discriminator():
    input_hazy = tf.keras.layers.Input(shape=(256, 256, 3))
    input_clear = tf.keras.layers.Input(shape=(256, 256, 3))

    # Concatenate the feature maps from the ResNet backbone
    inp = tf.keras.layers.concatenate([input_hazy, input_clear])
    init = RandomNormal(stddev=0.02, mean=0.0)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(inp)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)

    return tf.keras.Model(inputs=[input_hazy, input_clear], outputs=patch_out)
