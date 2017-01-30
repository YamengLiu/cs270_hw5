# svmClassifier.py
# ---------------

# adapted by Richard Guo, for CPS 270
# guo@cs.duke.edu

import util
import classificationMethod
from sklearn import svm, grid_search  # you can use scikit-learn as a module in this assignment. You are free to import more stuff from sklearn.
import numpy as np                      # you are also allowed to use the numerical module "numpy" (e.g. for computing mean and std)
from math import sqrt					# you may need this

class SupportVectorMachineClassifier(classificationMethod.ClassificationMethod):
  """
  The SVM classifier based on scikit-learn.
  """
  def __init__(self, legalLabels):
    self.type = "svm"
    self.clf=0
    print "\nSVM classifier working now...\n"

  def formattingData(self, data, features, mean=None, std=None):
    '''
    Convert data in the form of a list of dict to a n x p list of list object, 
    which is used by scikit-learn. n is the number of training samples and p is the number of features.
    And normalize data with given mean and std so that each feature has mean 0 and std 1.

    features: the list of features

    mean, std: both are dictionaries indexed by features. 
    If mean==None, mean is set to all zero; if std==None, std is set to all one.

    * Hint: 
    Normalizing each feature to have mean 0 and sd 1 can enhance the performance of SVM. 
    '''
    if mean==None:
      mean = dict((u, 0.0) for u in features)
    if std==None:
      std = dict((u, 1.0) for u in features)

    return [[(datum[u] - mean[u])/std[u] for u in features] for datum in data]


  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Fill out your code here to train a SVM classifier.

    To have your classifier *continue to live* after training, write something like "self.myclassifier".

    You may need to call formattingData(...) to convert trainingData to an appropriate form for scikit-learn.

    trainingData: A list [datum_1, datum_2, ...], with datum_i being i-th sample. 
                  Each datum_i is a dictionary in the form of {feature_k: value_k}

    trainingLabels: A list of class lables (integers 0 - 9), of the same length as trainingData

    validationData: validation set of data, of the same type as trainingData

    validationLabels: validation class labels, of the same tyoe as trainingLabels

    No returning value is needed in this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))
    "*** YOUR CODE HERE ***"
    #calculate mean and std of each dimension in trainingData
    self.mean={}
    self.std={}
    for key in self.features:
      aList=[]
      for datum in trainingData:
        aList.append(datum[key])
      bArray=np.array(aList)
      self.mean[key]=np.mean(bArray)    
      self.std[key]=np.std(bArray)
    #normalize trainingData
    x=[]
    for datum in trainingData:
      p=[]
      for key in self.features:
        if self.std[key]==0.0:
          p.append(0)
        else:
          p.append((datum[key]-self.mean[key])/self.std[key])
      x.append(p)
  
    y=trainingLabels
    #normalize validationData by using mean and std of trainingData
    x1=[]
    for datum in validationData:
      p=[]
      for key in self.features:
        if self.std[key]==0.0:
          p.append(0)
        else:
          p.append((datum[key]-self.mean[key])/self.std[key])
      x1.append(p)
    validationData=x1
    #optimize parameters and fit data
    parameters= [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
    self.clf=grid_search.GridSearchCV(svm.SVC(),parameters)
    self.clf.fit(x,y)
   
  def classify(self, testData):
    """
    Use the trained SVM classifier to predict the class labels of testData.

    testData: test dataset of the same type as trainingData in self.train(...)

    return: a list of predicted class labels.
    """

    "*** YOUR CODE HERE ***"
    #normalize testData by using mean and std of trainingData
    x1=[]
    x1=[]
    for datum in testData:
      p=[]
      for key in self.features:
        if self.std[key]==0.0:
          p.append(0)
        else:
          p.append((datum[key]-self.mean[key])/self.std[key])
      x1.append(p)
    testData=x1
    return self.clf.predict(testData)
    
    
