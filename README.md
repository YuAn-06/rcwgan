# RCWGAN
A CWGAN model with regression.
we proposed a novel virtual sample generation method embedding a deep neural network as a regressor into conditional Wasserstein generative adversarial networks with gradient penalty (rCWGAN). In rCWGAN, conditional variables are introduced making the training supervised and a dual training algorithm is specially designed. With the advanced structure and training algorithm, the model has powerful sample generation capabilities and can handle prediction problems of quality variable.

# Environment
Python 3.6

# Dependencies
* numpy
* pandas
* matplotlib
* tensorflow
* keras 

all packages need to be installed on a conda environment with python >= 3.0

# Acknowledgement
We appreciate efforts in https://github.com/eriklindernoren/Keras-GAN and in https://github.com/mkirchmeyer/ganRegression.
