from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import numpy as np
from sklearn.decomposition import PCA

#Fetching the dataset
import pandas as pd
import sys
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import StandardScaler

def generateImputedDataset(dataset):
    sys.setrecursionlimit(10000) #Increase the recursion limit of the OS

    # start the KNN training
    imputed_training=fast_knn(dataset1_train.values, k=30)
    imputed_dataset1 = pd.DataFrame(imputed_training)
    return imputed_dataset1

dataset1_train = pd.read_csv('./Dataset/TrainData1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training1 = generateImputedDataset(dataset1_train)
dataset1_labels = pd.read_csv('./Dataset/TrainLabel1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training1.insert(loc=len(imputed_training1.columns), column='label',value=dataset1_labels)
dataset1_test = pd.read_csv('./Dataset/TestData1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test1= generateImputedDataset(dataset1_test)

dataset2_train = pd.read_csv('./Dataset/TrainData2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training2 = generateImputedDataset(dataset2_train)
dataset2_labels = pd.read_csv('./Dataset/TrainLabel2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training2.insert(loc=len(imputed_training2.columns), column='label',value=dataset2_labels)
dataset2_test = pd.read_csv('./Dataset/TestData2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test2= generateImputedDataset(dataset2_test)

dataset3_train = pd.read_csv('./Dataset/TrainData3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training3 = generateImputedDataset(dataset3_train)
dataset3_labels = pd.read_csv('./Dataset/TrainLabel3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training3.insert(loc=len(imputed_training3.columns), column='label',value=dataset3_labels)
dataset3_test = pd.read_csv('./Dataset/TestData3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test3= generateImputedDataset(dataset3_test)


train, target = pd.DataFrame(dataset1_train), pd.DataFrame(dataset1_test)

