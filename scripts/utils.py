# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:39:15 2020

@author: Andrei
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt

#%matplotlib inline

def set_pretty_prints():
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.max_colwidth', 500)
  pd.set_option('display.width', 1000)

  np.set_printoptions(precision=2)
  np.set_printoptions(suppress=True)
  np.set_printoptions(threshold=np.inf)
  np.set_printoptions(linewidth=500)
  plt.style.use('ggplot')
  return

def load_dataset(dataset):
  DATASETS = {
    'imobiliare.ro' : '../data/imobiliare_bucuresti.csv',
    
    }
  if dataset not in DATASETS:
    raise ValueError("Uknown dataset '{}'".format(dataset))
  return pd.read_csv(DATASETS[dataset])
  
  
  
  
def load_sklearn_dataset(dataset_name, dev_slice=0.1, normalize=True):
  func_name = 'load_'+dataset_name
  if not hasattr(datasets, func_name):
    func_name = 'fetch_'+dataset_name
    if not hasattr(datasets, func_name):
      raise ValueError("Unknown dataset '{}'".format(func_name))
  func = getattr(datasets, func_name)
  print("Loading dataset '{}' using '{}' with normalize={}...".format(
      dataset_name, func.__name__, normalize))
  obj = func()
  df = pd.DataFrame(obj.data, columns=obj.feature_names)
  
  X = obj.data.astype(np.float32)
  y = obj.target.reshape((-1,1)).astype(np.float32)
  if normalize:
    X = (X - X.mean(axis=0)) / X.std(axis=0)    
  
  x_trn, x_ttt, y_trn, y_ttt = train_test_split(X, y, test_size=dev_slice * 2)
  
  x_dev, x_test, y_dev, y_test = train_test_split(x_ttt, y_ttt, test_size=0.5)
  
  dct = OrderedDict({
      'x_full'  : X,
      'y_full'  : y,
      'x_train' : x_trn,
      'y_train' : y_trn,
      'x_dev'   : x_dev,
      'y_dev'   : y_dev,
      'x_test'  : x_test,
      'y_test'  : y_test,
      'df'      : df,
      'desc'    : obj.DESCR,
      })
  Dataset = namedtuple('Dataset', list(dct.keys()))
  print("Data loaded:")
  for key in dct:
    print("  {:<10} {}".format(
        key,
        dct[key].shape if type(dct[key]) in [np.ndarray, pd.DataFrame] else '"'+dct[key][:100].replace('\n','')+'"'))
  return Dataset(**dct)

  

if __name__ == '__main__':
  pass
#  boston = load_dataset('boston')
#  calif = load_dataset('california_housing')
#  diabt = load_dataset('diabetes')
#  bc = load_dataset('breast_cancer')
  

  