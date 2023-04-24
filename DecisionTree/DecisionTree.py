import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
#Disable panda warning about chained assignments modifying the original table
pd.options.mode.chained_assignment = None

################################
# Loading Data
################################
#Get the data files from the parent directory
try:
    fileName = os.path.join(os.getcwd(), "..", "adult.filtered")
    fileNameTest = os.path.join(os.getcwd(), "..", "adult.filteredTest")

    #load the data into a pandas DataFrame
    columnNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    data = pd.read_csv(fileName, header=None, names=columnNames)
    featureColumns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    #Create a separate Table for the X's, and for the Y's
    xTrain = data[featureColumns]
    yTrain = data.label

    #Do the same for the test data
    dataTest = pd.read_csv(fileNameTest, header=None, names=columnNames)
    xTest = dataTest[featureColumns]
    yTest = dataTest.label
except (IOError, OSError, FileNotFoundError) as e:
    print("Failed to load data, exiting.")
    print(os.getcwd())
    exit()


################################
# Data Pre-processing
################################
#Some values for native-country that appear in the training data do not appear
# in the test data, so make sure the test data has the same categories
# for native-country as training data by setting it the same
# for both sets of data
possible_categories = [" United-States", " Cambodia", " England", " Puerto-Rico", " Canada", " Germany", " Outlying-US(Guam-USVI-etc)", " India", " Japan", " Greece", " South", " China", " Cuba", " Iran", " Honduras", " Philippines", " Italy", " Poland", " Jamaica", " Vietnam", " Mexico", " Portugal", " Ireland", " France", " Dominican-Republic", " Laos", " Ecuador", " Taiwan", " Haiti", " Columbia", " Hungary", " Guatemala", " Nicaragua", " Scotland", " Thailand", " Yugoslavia", " El-Salvador", " Trinadad&Tobago", " Peru", " Hong", " Holand-Netherlands"]
dtype = pd.CategoricalDtype(categories=possible_categories)
xTrain["native-country"] = pd.Series(xTrain["native-country"], dtype=dtype)
xTest["native-country"] = pd.Series(xTest["native-country"], dtype=dtype)

#Scikit-Learn's Decision Tree algorithm does not work on categorical data, so we use panda's get_dummies method
# to One-hot encode the data. This splits each column of categorical data into multiple columns for each possible
# value of the category. It will make the value of the column 1 for the data entry's value for that category, and
# 0 for all other columns related to that category.
# This is done for each categorical data type in the training and testing datasets
categoricalValues = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
for category in categoricalValues:
    xTrain = pd.get_dummies(xTrain,prefix=[category], columns=[category], drop_first=True)
    xTest = pd.get_dummies(xTest,prefix=[category], columns=[category], drop_first=True)

#Convert the Y labels into binary values
# 0 = <=50K
# 1 = >50K
yTrain = LabelBinarizer().fit_transform(yTrain)
yTest = LabelBinarizer().fit_transform(yTest)


################################
# Model training
################################
#Create and train a decision tree to the training data
#Test limiting the max depth from 1 to 60, saving all the accuracies along the way to plot.
#Save the best accuracy to be printed at the end.
maxDepths = []
accuracies = []
bestDepth = 0
bestAccuracy = 0
for i in range(1, 61):
    DecisionModel = DecisionTreeClassifier(max_depth=i)
    DecisionModel = DecisionModel.fit(xTrain, yTrain)
    
    yPred = DecisionModel.predict(xTest)
    accuracy = metrics.accuracy_score(yTest, yPred)
    print("Max Depth: {:2d}  Accuracy: {:.8f}".format(i, accuracy))
    
    #Append to the lists to be graphed
    maxDepths.append(i)
    accuracies.append(accuracy)
    
    #If the accuracy of the current depth was better than the previous best, update accordingly
    if accuracy > bestAccuracy:
        bestDepth = i
        bestAccuracy = accuracy


################################
# Predictions and Accuracy
################################
#Use the best depth found above for the accuracy to be printed
DecisionModel = DecisionTreeClassifier(max_depth=bestDepth)
DecisionModel = DecisionModel.fit(xTrain, yTrain)

#Create a prediction of the results of the test data, based on the model created above
yPred = DecisionModel.predict(xTest)

#Compare the predicted results to the actual results in the test data to find the accuracy of the model
print()
print("Best Max Depth: " + str(bestDepth))
print("Best Accuracy: " + str(metrics.accuracy_score(yTest, yPred)))

#Plot depth vs. accuracy
plt.plot(maxDepths, accuracies, c="green", label="Accuracy")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.scatter(bestDepth, bestAccuracy, marker="o", color="black", linewidths=0.5)
plt.text(bestDepth, bestAccuracy, "({}, {})".format(bestDepth, bestAccuracy))
plt.legend(loc="best")
plt.title("Maximum Depth vs. Decision Tree Accuracy")
print("\nClose the plt window and the decision tree will be saved to an SVG file.")
plt.show()


################################
# Graphing the Decision Tree
################################
#Create an SVG image file of the decision tree
print("Creating decision tree image...")
print("(May take some time)")
outputFile = "DecisionTreeOutput.svg"
dot_data = StringIO()
export_graphviz(DecisionModel, out_file=dot_data, filled=True, rounded=True, special_characters=True, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_svg(outputFile)
Image(graph.create_svg())
print("Image saved to: " + outputFile)

