from __future__ import print_function,division
import numpy as np
from keras.layers import Input
from keras import  Model
from keras.optimizers import Adam,SGD
from network import build_generator,build_discriminator,build_regressor
import keras.backend as K
from keras.layers.merge import add
import tensorflow as tf
from functools import partial
from sklearn.model_selection import KFold
tf.compat.v1.disable_eager_execution()


class rcwgan():
    def __init__(self,exp_config):
        if exp_config.model.optim_gen =="Adam":
            self.optimizer_gen = Adam(learning_rate= exp_config.model.lr_gen,
                                      beta_1=exp_config.model.beta1,beta_2=exp_config.model.beta2)
        else:
            self.optimizer_gen = SGD(exp_config.model.lr_gen, decay=exp_config.model.decay_gen)

        if exp_config.model.optim_disc =="Adam":
            self.optimizer_disc = Adam(learning_rate= exp_config.model.lr_disc,
                                       beta_1=exp_config.model.beta1,beta_2=exp_config.model.beta2)
        else:
            self.optimizer_disc = SGD(exp_config.model.lr_disc, decay=exp_config.model.decay_disc)

        if exp_config.model.optim_reg == 'Adam':
            self.optimizer_reg = Adam(learning_rate= exp_config.model.lr_reg,
                                      )
        else:
            self.optimizer_reg = SGD(exp_config.model.lr_reg, decay=exp_config.model.decay_reg)

        self.activation  =  exp_config.model.activation
        # self.seed = exp_config.model.random_seed
        self.scenario = exp_config.dataset.scenario
        # self.n_samplong = exp_config.training.n_sampling
        print(self.scenario)

        if self.scenario == "pta":
            self.x_input_size = 17
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 2
            self.batch_size = 35
            self.critic_num = exp_config.training.critic_num
        elif self.scenario == 'standard_data':
            self.x_input_size = 1
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 1
            self.batch_size = exp_config.training.batch_size
            self.critic_num = exp_config.training.critic_num
        else:
            self.x_input_size = 17
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 2
            self.batch_size = 35


        if exp_config.model.architecture is not None:
            self.architecture = exp_config.model.architecutre


        self.generator = build_generator(self)
        self.discriminator = build_discriminator(self)
        self.regressor = build_regressor(self)

        self.regressor.compile(optimizer=self.optimizer_reg, loss='mse')
        self.generator.trainable = False

        x = Input(shape=(self.x_input_size,), name="real_x")
        y = Input(shape=(self.y_input_size,), name="label")
        fake_y = Input(shape=(self.y_input_size,), name="fake_label")
        z = Input(shape=(self.z_input_size,))

        fake_x = self.generator([z, fake_y])

        fake = self.discriminator([fake_x, fake_y])
        valid = self.discriminator([x, y])

        interpolated_data = self.RandomWeightAverage([x, fake_x])
        interpolated_value = self.discriminator([interpolated_data, y])

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_data, in_label=y)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = Model(inputs=[x, z, y, fake_y],
                                         outputs=[valid, fake, interpolated_value])
        self.discriminator_model.compile(optimizer=self.optimizer_disc,
                                         loss=[self.wassertein_loss, self.wassertein_loss, partial_gp_loss],
                                         loss_weights=[1, 1, 10], experimental_run_tf_function=False)

        self.discriminator.trainable = False
        self.generator.trainable = True

        z_gen = Input(shape=(self.z_input_size,), name="z_gen")
        label_gen = Input(shape=(self.y_input_size,), name="label_gen")
        g_z = self.generator([z_gen, label_gen])
        valid = self.discriminator([g_z, label_gen])
        self.generator_model = Model([z_gen, label_gen], valid)
        self.generator_model.compile(optimizer=self.optimizer_gen, loss=self.wassertein_loss)

        # print(self.generator_model.summary())
        # print(self.discriminator_model.summary())
        # print(self.regressor.summary())

    def RandomWeightAverage(self,inputs):
        alpha = K.random_uniform(shape=(self.batch_size,1),minval=0,maxval=1)
        inter = add([alpha*inputs[0],(1-alpha)*inputs[1]])
        return inter

    def wassertein_loss(self, y_true, y_pred):
        return  K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, in_label):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([averaged_samples, in_label])
            y_pred = self.discriminator([averaged_samples, in_label])
            gradient = tape.gradient(y_pred, [averaged_samples, in_label])
            gradient = tf.concat([gradient[0], gradient[1]], axis=-1)
            gradient_L2_norm = tf.norm(gradient, ord=2, axis=1)
            gradient_penalty = K.square(gradient_L2_norm - 1)
        return gradient_penalty


    def train(self,x_train,y_train,epochs):
        print(self.batch_size)
        valid = -np.ones((self.batch_size,1))
        fake = np.ones((self.batch_size,1))
        dummy = np.zeros((self.batch_size,1))
        dLossErr = np.zeros([epochs,1])
        gLossErr = np.zeros([epochs,1])
        rLossErr1 = np.zeros([epochs,1])
        rLossErr2 = np.zeros([epochs,1])
        print(epochs)
        for epoch in range(epochs):
            for _ in range(self.critic_num):
                idx = np.random.randint(0, x_train.shape[0], size=self.batch_size)
                x = x_train[idx]
                y = y_train[idx]
                # noise = tf.random.normal(shape=(self.batch_size,self.z_input_size))
                noise = np.random.normal(0,1,(self.batch_size,self.z_input_size))
                # z_g = self.generator.predict_on_batch([noise,y])
                r_loss1 = self.regressor.train_on_batch(x, y)
                z_g = self.generator.predict([noise,y])
                # r_label = self.regressor.predict_on_batch(z_g)
                r_label = self.regressor.predict(z_g)
                # r_loss1 = self.regressor.train_on_batch(x,y)
                # r_loss2 = self.regressor.train_on_batch(z_g,y)
                d_loss = self.discriminator_model.train_on_batch([x,noise,y,r_label],[valid,fake,dummy])

            r_loss2 = self.regressor.train_on_batch(z_g, y)
            g_loss = self.generator_model.train_on_batch([noise,y],valid)
            # dLossErr[epoch] = d_loss[0]
            # gLossErr[epoch] = g_loss
            # rLossErr1[epoch] = r_loss1
            # rLossErr2[epoch] = r_loss2
            print("====[Epoch: %d/%d] [D loss: %f] [G loss: %f],[R loss1:  %f],[R loss2:  %f]" % (epoch+1,epochs, d_loss[0], g_loss,r_loss1,r_loss2))



    def predict(self,x_test):
        y = self.regressor.predict(x_test)

        return y



