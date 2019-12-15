import numpy as np
import pandas as pd
from random import *

from sklearn import datasets
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier



PCA_COMPONENT_NO = 11
NO_OF_SPLITS = [3, 5, 10]
SEED = randint(1, 9999999)


classifier_dictionary_list = [
    ["LR_C0", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1)) , "logistic regression C=1"],
    ["LR_C1", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e1)), "logistic regression C=1e1"],
    ["LR_C2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e2)), "logistic regression C=1e2"],
    ["LR_C3", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e3)), "logistic regression C=1e3"],
    ["LR_C4",  make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e4)), "logistic regression C=1e4"],
    ["LR_C5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e5)), "logistic regression C=1e5"],
    ["LR_C6", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e6)), "logistic regression C=1e6"],
    ["RF20", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=20)), "Random Forest 20 estimator"],
    ["RF10", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=10)), "Random Forest 10 estimator"],
    ["RF5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=5)), "Random Forest 5 estimator"],
    ["RF2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=2)), "Random Forest 2 estimator"],
    ["DT", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), tree.DecisionTreeClassifier()), "Random Forest 10 estimator"],
    ["SVM_RBF", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='rbf', C = 1)), "svm.SVC kernel=linear"],
    ["SVM_LINEAR", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='linear', C = 1)), "svm.SVC kernel=linear"],
    ["SVM_SIGMOID", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='sigmoid', C = 1)), "svm.SVC kernel=linear"],
    ["KNN1", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=1)), "KNN k = 1"],
    ["KNN2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=2)), "KNN k = 2"],
    ["KNN5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=5)), "KNN k = 5"],
    ["KNN10", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=10)), "KNN k = 10"],
    ["KNN20", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=20)), "KNN k = 20"],
    ["KNN50", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=50)), "KNN k = 50"] 
    
]



def readDataset1():
    
	trainDataFile = "../dataset/ImputedTrainData1.txt"
	trainLabelFile = "../dataset/TrainLabel1.txt"
	testDataFile = "../dataset/ImputedTestData1.txt"

	trainData = genfromtxt(trainDataFile, delimiter="\t")
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter="\t")
	
	return trainData, trainLabel, testData




def readDataset2():
    
	trainDataFile = "../dataset/TrainData2.txt"
	trainLabelFile = "../dataset/TrainLabel2.txt"
	testDataFile = "../dataset/TestData2.txt"
	trainData = pd.read_csv(trainDataFile,sep='   ', header=None,index_col=False,engine="python")
	trainLabel = pd.read_csv(trainLabelFile,sep='\n', header=None,index_col=False,engine="python",skipinitialspace=True)
	testData = pd.read_csv(testDataFile,sep='   ', header=None,index_col=False,engine="python")
	
	return trainData, trainLabel, testData	
	
	
	
def readDataset3():
    
	trainDataFile = "../dataset/ImputedTrainData3.txt"
	trainLabelFile = "../dataset/TrainLabel3.txt"
	testDataFile = "../dataset/TestData3.txt"

	trainData = genfromtxt(trainDataFile, delimiter='\t')
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter=',')
	
	return trainData, trainLabel, testData
	


def readDataset4():
    
	trainDataFile = "../dataset/TrainData4.txt"
	trainLabelFile = "../dataset/TrainLabel4.txt"
	testDataFile = "../dataset/TestData4.txt"
	trainData = pd.read_csv(trainDataFile,sep='   |  ', header=None,index_col=False,engine="python")
	trainLabel = pd.read_csv(trainLabelFile,sep='\n', header=None,index_col=False,engine="python",skipinitialspace=True)
	testData = pd.read_csv(testDataFile,sep='   |  ', header=None,index_col=False,engine="python")
	
	return trainData, trainLabel, testData

	
	
def readDataset5():
    
	trainDataFile = "../dataset/TrainData5.txt"
	trainLabelFile = "../dataset/TrainLabel5.txt"
	testDataFile = "../dataset/TestData5.txt"

	trainData = genfromtxt(trainDataFile, delimiter='   ')
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter='   ')
	
	return trainData, trainLabel, testData

	
	
def readDataset6():
    
	trainDataFile = "../dataset/TrainData6.txt"
	trainLabelFile = "../dataset/TrainLabel6.txt"
	testDataFile = "../dataset/TestData6.txt"
	trainData = pd.read_csv(trainDataFile,sep='   |  ', header=None,index_col=False,engine="python")
	trainLabel = pd.read_csv(trainLabelFile,sep='\n', header=None,index_col=False,engine="python",skipinitialspace=True)
	testData = pd.read_csv(testDataFile,sep='   |  ', header=None,index_col=False,engine="python")

	return trainData, trainLabel, testData


	
def write_data(filename, array):

    with open(filename, "w") as f:
        for l in array:
            f.write(str(int(l)) + "\n")




def getClassifier(name):
    for classifier in classifier_dictionary_list:
        if classifier[0] == name:
            return classifier

			


def trainClassifier(trainData, trainLabel, testData, predictedLabelFile):
	givenTestFeatures = testData	
	accuracyVector = []
	
	for i in classifier_dictionary_list:	
		bestAccuracyKFold = 0
		try:			
			for k in NO_OF_SPLITS:
				kFold = StratifiedKFold(n_splits = k, shuffle = True, random_state = SEED)
				
				classifier = i[1]
				scores = cross_val_score(classifier, trainData, trainLabel, cv = kFold, scoring='accuracy')
				print(scores)      				
					
				print("Accuracy with scaled data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
				
				bestAccuracyKFold = max(scores.mean(), bestAccuracyKFold)								
						
			
			#print(bestAccuracyKFold)			
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

		accuracyVector.append(bestAccuracyKFold)
		
			
	#np.where(a==a.max())
	print("accuracyVector :: ")
	print(accuracyVector)
	
	print("highest accurate classifier :: ")
	
	print(classifier_dictionary_list[np.argmax(accuracyVector, axis=0)][0])
	print("highest accuracy :: ")
	print(np.amax(accuracyVector, axis=0))
	mostAccurateClassifier = classifier_dictionary_list[np.argmax(accuracyVector, axis=0)][1]
	
	mostAccurateClassifier.fit(trainData, trainLabel)
	preditedLabel = mostAccurateClassifier.predict(givenTestFeatures)
	data = preditedLabel.reshape(preditedLabel.shape[0], 1)
	write_data(predictedData, data)
		
	return mostAccurateClassifier
	
	

trainData, trainLabel, testData = readDataset1()	
predictedData = "../result/RayClassification1.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)

trainData, trainLabel, testData = readDataset2()	
predictedData = "../result/RayClassification2.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)


trainData, trainLabel, testData = readDataset3()	
predictedData = "../result/RayClassification3.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)

trainData, trainLabel, testData = readDataset4()	
predictedData = "../result/RayClassification4.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)

trainData, trainLabel, testData = readDataset5()	
predictedData = "../result/RayClassification5.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)

trainData, trainLabel, testData = readDataset6()	
predictedData = "../result/RayClassification6.txt"
trainClassifier(trainData, trainLabel, testData, predictedData)




