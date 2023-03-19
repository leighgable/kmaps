from sklearn import datasets
from sklearn.utils import Bunch

import pickle


def get_iris_data():
    return datasets.load_iris() # sklearn "bunch" obj
    
def save(path, filename, obj):    
    with open(path+'/'+filename, 'wb') as bunch:
        print(path+'/'+filename)
        pickle.dump(obj, bunch, protocol=pickle.HIGHEST_PROTOCOL)

def load_it(path, file):            
    with open(path+'/'+file, 'rb') as bunch:
        print(path+'/'+file)
        return pickle.load(bunch)

