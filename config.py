import typing

class ModelConfig(typing.NamedTuple):
    architecture : int = None
    lr_gen: float = 0.0001
    lr_disc : float = 0.0004
    lr_reg : float  = 0.0001
    optim_gen : str = "Adam"
    optim_disc : str = 'Adam'
    optim_reg : str = 'Adam'
    decay_gen : float = 0
    decay_disc : float = 0
    decay_reg: float = 0
    beta1 : float = 0
    beta2 : float = 0
    z_input_size : int = 10
    activation : str = "elu"

class TrainingConfig(typing.NamedTuple):
    epochs: int = 100
    batch_size : int =  35
    critic_num : int = 3

class DatasetConfig(typing.NamedTuple):
    scenario : str = "standard_data"
    n_instance : int = 1000

class Config(typing.NamedTuple):
    model : ModelConfig
    dataset : DatasetConfig
    training : TrainingConfig

