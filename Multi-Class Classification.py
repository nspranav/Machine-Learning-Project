from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import pandas as pd
import sys
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model
from random import *

def generateImputedDataset(dataset):
    sys.setrecursionlimit(10000) #Increase the recursion limit of the OS

    # start the KNN training
    imputed_training=fast_knn(dataset.values, k=30)
    imputed_dataset1 = pd.DataFrame(imputed_training)
    return imputed_dataset1

scaler = MinMaxScaler(feature_range=[0, 1])
NO_OF_SPLITS = [3, 5, 10]

classifiers = {
    "LR_C0": [LogisticRegression(C=1) , "logistic regression C=1"],
    "LR_C2": [LogisticRegression(C=1e2), "logistic regression C=100"],
    "LR_C1": [LogisticRegression(C=1e1), "logistic regression C=1000"],
    "LR_C3": [LogisticRegression(C=1e3), "logistic regression C=1000"],
    "LR_C4": [ LogisticRegression(C=1e4), "logistic regression C=10000"],
    "LR_C5": [LogisticRegression(C=1e5), "logistic regression C=1000000"],
    "LR_C6": [LogisticRegression(C=1e6), "logistic regression C=1000000"],
    "RF20": [RandomForestClassifier(n_estimators=20), "Random Forest 20 estimator"],
    "RF10": [RandomForestClassifier(n_estimators=10), "Random Forest 10 estimator"],
    "RF5": [RandomForestClassifier(n_estimators=5), "Random Forest 5 estimator"],
    "RF2": [RandomForestClassifier(n_estimators=2), "Random Forest 2 estimator"],
    "DT": [tree.DecisionTreeClassifier(), "Random Forest 10 estimator"],
    "SVM_RBF": [svm.SVC(kernel='rbf', C = 1), "svm.SVC kernel=linear"],
    "SVM_LINEAR": [svm.SVC(kernel='linear', C = 1), "svm.SVC kernel=linear"],
    "SVM_SIGMOID": [svm.SVC(kernel='sigmoid', C = 1), "svm.SVC kernel=linear"],
    "KNN1": [KNeighborsClassifier(n_neighbors=1), "KNN k = 1"],
    "KNN2": [KNeighborsClassifier(n_neighbors=2), "KNN k = 2"],
    "KNN5": [KNeighborsClassifier(n_neighbors=5), "KNN k = 5"],
    "KNN10": [KNeighborsClassifier(n_neighbors=10), "KNN k = 10"],
    "KNN20": [KNeighborsClassifier(n_neighbors=20), "KNN k = 20"],
    "KNN50": [KNeighborsClassifier(n_neighbors=50), "KNN k = 50"] 
}

def trainClassifier(trainData, trainLabel, testData):
	givenTestFeatures = testData	
	accuracy = []
	
	for i in classifiers:	
		bestAccuracyScore = 0
		try:			
			for k in NO_OF_SPLITS:
				kFold = StratifiedKFold(n_splits = k, shuffle = True, random_state = randint(1, 9999999))
				
				classifier = classifiers[i][0]
				scores = cross_val_score(classifier, trainData, trainLabel, cv = kFold, scoring='accuracy')
				print(scores)      				
					
				print("Accuracy with scaled data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                
                if scores.mean() > bestAccuracyScore:



                								
						
			
			print(bestAccuracyKFold)			
		except IOError:
			print('An error occurred trying to read the file.')

		except ValueError:
			print('Non-numeric data found in the file.')

		except ImportError:
			print ("NO module found")

		except EOFError:
			print('EOFError')

		except KeyboardInterrupt:
			print('You cancelled the operation.')

		accuracy.append(bestAccuracyKFold)
		
			
	#np.where(a==a.max())
	print("accuracyVector :: ")
	print(accuracyVector)
	
	print("highest accurate classifier :: ")
	
	print(classifiers[np.argmax(accuracyVector, axis=0)][0])
	print("highest accuracy :: ")
	print(np.amax(accuracyVector, axis=0))
	mostAccurateClassifier = classifiers[np.argmax(accuracyVector, axis=0)][1]
	
	mostAccurateClassifier.fit(trainData, trainLabel)
	preditedLabel = mostAccurateClassifier.predict(givenTestFeatures)
	data = preditedLabel.reshape(preditedLabel.shape[0], 1)
	print(data)
		
	return mostAccurateClassifier
	

dataset1_train = pd.read_csv('./Dataset/TrainData1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training1 = generateImputedDataset(dataset1_train)
dataset1_labels = pd.read_csv('./Dataset/TrainLabel1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
#imputed_training1.insert(loc=len(imputed_training1.columns), column='label',value=dataset1_labels)
dataset1_test = pd.read_csv('./Dataset/TestData1.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test1= generateImputedDataset(dataset1_test)

#normalizing the data for attaining the maximum variance
data1_rescaled = scaler.fit_transform(imputed_training1)
data1_rescaled_test = scaler.fit_transform(imputed_test1)


# performing the PCA 
pca = PCA().fit(data1_rescaled)#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()

#performing the PCA by selecting the maximum components
pca = PCA(n_components=148)
dataset1_train_reduced = pca.fit_transform(data1_rescaled)
dataset1_test_reduced = pca.fit_transform(data1_rescaled_test)

dataset2_train = pd.read_csv('./Dataset/TrainData2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training2 = generateImputedDataset(dataset2_train)
dataset2_labels = pd.read_csv('./Dataset/TrainLabel2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
#imputed_training2.insert(loc=len(imputed_training2.columns), column='label',value=dataset2_labels)
dataset2_test = pd.read_csv('./Dataset/TestData2.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test2= generateImputedDataset(dataset2_test)

dataset3_train = pd.read_csv('./Dataset/TrainData3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_training3 = generateImputedDataset(dataset3_train)
dataset3_labels = pd.read_csv('./Dataset/TrainLabel3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
#imputed_training3.insert(loc=len(imputed_training3.columns), column='label',value=dataset3_labels)
dataset3_test = pd.read_csv('./Dataset/TestData3.txt',header=None, delimiter='\t',na_values= ['1.00000000000000e+99'])
imputed_test3= generateImputedDataset(dataset3_test)
