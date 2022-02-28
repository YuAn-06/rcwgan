import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, Concatenate,BatchNormalization

tf.compat.v1.disable_eager_execution()

"""Generator"""


def build_generator(network):
    print(network.activation)
    random_normal = keras.initializers.RandomNormal

    if network.activation == 'linear':
        activation = 'linear'
        k_initial = keras.initializers.RandomUniform(seed=1998)
    elif network.activation == 'elu':
        activation = 'elu'
        k_initial = keras.initializers.he_normal(seed=1998)
    elif network.activation == 'relu':
        activation = 'relu'
        k_initial = keras.initializers.he_uniform(seed=1998)
    elif network.activation == 'LeakyReLu':
        activation = LeakyReLU(alpha=0.2)
        k_initial = keras.initializers.he_normal(seed=1998)
    else:
        raise NotImplementedError("Activation not recognized")
    """G for standard dataset"""
    if network.architecture == 1:
        label = Input(shape=(network.y_input_size,), dtype=float, name="Generator_input_y")
        # label_output = Dense(5,dtype=float,name='G_layer1')(label)
        noise = Input(shape=(network.z_input_size,), dtype=float, name="Generator_input_noise")
        # noise_output = Dense(5, dtype=float, name='G_layer2')(label)

        output = Concatenate()([noise,label])
        # output = Concatenate()([noise_output, label_output])
        output = Dense(64, activation='elu', kernel_initializer=k_initial)(output)
        output = Dense(64, activation='elu', kernel_initializer=k_initial)(output)
        output = Dense(64, activation='elu', kernel_initializer=k_initial)(output)

        gz = Dense(network.x_input_size, activation="linear")(output)
        model = Model(inputs=[noise, label], outputs=gz)
        return model
    elif network.architecture == 2:
        label = Input(shape=(network.y_input_size,), dtype=float, name="Generator_input_y")
        noise = Input(shape=(network.z_input_size,), dtype=float, name="Generator_input_noise")
        output = Concatenate()([noise, label])
        output = Dense(512, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(256, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(128, activation=activation, kernel_initializer=k_initial)(output)
        gz = Dense(units=network.x_input_size, activation="linear")(output)
        model = Model(inputs=[noise, label], outputs=gz)
        return model
    else:
        raise NotImplementedError("Achitecture dose not exits")


"""Discriminator"""


def build_discriminator(network):
    random_uniform = keras.initializers.RandomUniform

    if network.activation == 'linear':
        activation = 'linear'
        k_initial = keras.initializers.RandomUniform(seed=1992)
    elif network.activation == 'elu':
        activation = 'elu'
        k_initial = keras.initializers.he_normal(seed=1992)
    elif network.activation == 'relu':
        activation = 'relu'
        k_initial = keras.initializers.he_uniform(seed=1992)
    else:
        raise NotImplementedError("Activation not recognized")
    """D for standard dataset"""
    if network.architecture == 1:
        label = Input(shape=(network.y_input_size,), dtype=float, name="Discriminator_input_y")
        # label_output = Dense(10,activation=activation)(label)
        d_x = Input(shape=(network.x_input_size,), dtype=float, name="Discriminator_input_x")
        # d_x_output = Dense(10,activation=activation)(d_x)
        # d_input = Concatenate()([d_x_output, label_output])
        d_input = Concatenate()([d_x, label])
        output = Dense(64, activation=activation, kernel_initializer=k_initial)(d_input)
        output = Dense(64, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(64, activation=activation, kernel_initializer=k_initial)(output)
        valid = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(output)
        model = Model(inputs=[d_x, label], outputs=valid)

    elif network.architecture == 2:
        label = Input(shape=(network.y_input_size,), dtype=float, name="Discriminator_input_y")
        d_x = Input(shape=(network.x_input_size,), dtype=float, name="Discriminator_input_x")
        output = Concatenate()([d_x, label])

        output = Dense(256, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(256, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(128, activation=activation, kernel_initializer=k_initial)(output)
        valid = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(output)
        model = Model(inputs=[d_x, label], outputs=valid)

    else:
        raise NotImplementedError("Achitecture dose not exits")
    return model


"""Regressor"""


def build_regressor(network):
    # seed = network.seed
    random_normal = keras.initializers.RandomNormal

    if network.activation == 'linear':
        activation = 'linear'
        k_initial = keras.initializers.RandomUniform(seed=1992)
    elif network.activation == 'elu':
        activation = 'elu'
        k_initial = keras.initializers.he_normal(seed=1992)
    elif network.activation == 'relu':
        activation = 'relu'
        k_initial = keras.initializers.he_uniform(seed=1992)
    else:
        raise NotImplementedError("Activation not recognized")
    """R for standard dataset"""
    if network.architecture == 1:
        x = Input(shape=(network.x_input_size,), dtype=float, name="Regressor_input")

        output = Dense(64, activation=activation, kernel_initializer=k_initial)(x)
        output = Dense(64, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(64, activation=activation, kernel_initializer=k_initial)(output)
        output = Dense(network.y_input_size, activation="linear")(output)
        model = Model(inputs=x, outputs=output)

    elif network.architecture == 2:
        print("x_input_size = ", network.x_input_size)
        r_x = Input(shape=(network.x_input_size,), dtype=float, name="Regressor_input")
        output = Dense(256, activation="relu", kernel_initializer=keras.initializers.he_uniform)(r_x)
        output = Dense(128, activation="relu", kernel_initializer=keras.initializers.he_uniform)(output)
        output = Dense(64, activation="relu", kernel_initializer=keras.initializers.he_uniform)(output)
        pre_y = Dense(units=network.y_input_size, activation="linear")(output)
        model = Model(inputs=r_x, outputs=pre_y)
    else:
        raise NotImplementedError("Achitecture dose not exits")
    return model
