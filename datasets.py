import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1998)
def gen_data_standard(n_instance):
    def _randrange(n,vmin,vmax):
        return (vmax-vmin)*np.random.rand(n,)+vmin


    x1 = _randrange(n_instance,0,1)
    x2 = _randrange(n_instance,0,1)

    scale = 1 #noise scale
    noise = np.random.normal(0,0.0025,n_instance)

   
    X = x1
    
    y = _standard_sin(x1)
    y = y.reshape((n_instance,1))
    return X,y

def _standard_sin(x1):
    # y = (1.3356 * (1.5 * (1 - x1))
    #      + (np.exp(2 * x1 - 1) * np.sin(3 * np.pi * (x1 - 0.6) ** 2))
    #      + (np.exp(3 * (x2 - 0.5)) * np.sin(4 * np.pi * (x2 - 0.9) ** 2)))
    # y = x1 + 0.2*np.sin(20*x1)+np.random.uniform(0,0.05)
    y = x1 + 0.2 * np.sin(20 * x1)
    # y = (np.e**(x1-1)) * np.sin(4 * np.pi * (x1 - 0.6) ** 2)
    return y




def get_dataset(n_instance = 1000, scenario="pta",seed=1):
    if scenario =="pta":
        data = pd.read_excel('PTA_DATA.xlsx', header=None)
        data = data.values
        x = data[:, :17]
        y = data[:, 17]
     
        return x,y
    elif scenario == "standard_data":
        x,y = gen_data_standard(n_instance=n_instance)
        x_train, x_test = train_test_split(x, test_size=0.3, random_state=1)
        y_train, y_test = train_test_split(y, test_size=0.3, random_state=1)

        return x_train, y_train.reshape(-1, 1), x_test, y_test.reshape(-1, 1)

    else:
        raise NotImplementedError('Dataset does not exist')


