import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import OneHotEncoder


fileName = os.path.join(os.getcwd(), "..", "adult.filtered")
fileNameTest = os.path.join(os.getcwd(), "..", "adult.filteredTest")


columnNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
data = pd.read_csv(fileName, header=None, names=columnNames)

featureColumns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
xTrain = data[featureColumns]
yTrain = data.label

dataTest = pd.read_csv(fileNameTest, header=None, names=columnNames)

xTest = dataTest[featureColumns]
yTest = dataTest.label




#clf = DecisionTreeClassifier()
#clf = clf.fit(xTrain, yTrain)

#yPred = clf.predict(xTest)

#print("Accuracy:", metrics.accuracy_score(yTest, yPred))