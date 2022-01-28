# Copyright (C) 2021 #
# @Time    : 2022/1/28 10:31
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : run.py
# @Software: PyCharm
import importlib
import  datasets,metrics,config, plotting,network,rcwgan_model
import os
import numpy as np
import random
import  tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
from rcwgan_model import rcwgan
importlib.reload(network)
importlib.reload(datasets)
importlib.reload(metrics)
importlib.reload(config)
importlib.reload(rcwgan_model)
import pandas as pd

importlib.reload(plotting)
dataset_config = config.DatasetConfig(scenario='pta')

assert (dataset_config.scenario == 'pta'
        or dataset_config.scenario == 'standard_data')

fig_dir = f"../figures/{dataset_config.scenario} "
try:
    os.mkdir(fig_dir)
    print(f"Directory {fig_dir} created")
except FileExistsError:
    print(f"Directory {fig_dir} already exists replacing files")

exp_config = config.Config(model=config.ModelConfig(lr_gen=0.0001,lr_disc=0.0003,lr_reg=0.0005,activation="elu",
                                                    optim_gen='Adam',optim_disc='Adam',optim_reg='Adam',z_input_size=10,beta2=0.9,beta1=0),
                           training=config.TrainingConfig(epochs=1200,batch_size=35,critic_num=5),
                           dataset=config.DatasetConfig(scenario='pta')
                           )




x_train,y_train,x_test,y_test = datasets.get_dataset(scenario=exp_config.dataset.scenario
                                                                       seed=exp_config.model.random_seed)
x,y = datasets.get_dataset(scenario=exp_config.dataset.scenario)



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y.reshape(-1, 1))
x_train, x_test = train_test_split(x, test_size=0.3, random_state=1)
y_train, y_test = train_test_split(y, test_size=0.3, random_state=1)



x_train = x_scaler.fit_transform(x_train)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
x_test = x_scaler.transform(x_test)


import datetime

now = datetime.datetime.now()
model= rcwgan_model.rcwgan(exp_config)


"""
train
"""
model.train(x_train,y_train,exp_config.training.epochs)



"""
predict
"""
y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)


from metrics import RMSE,MAE
rmse = RMSE(y_test,y_pred)
mae = MAE(y_test,y_pred)

plt.plot(y_pred,'go-',c='b',label='test')
plt.plot(y_test,'ro-',c='orange',label='true')
plt.legend()
plt.show()

print("rmse:" ,rmse)
print("mae:",mae)
print(f"Training Time: {((datetime.datetime.now()-now).seconds)}")


