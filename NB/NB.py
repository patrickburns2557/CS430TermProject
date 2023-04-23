from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# input filenames
train_file = '../adult.filtered'
test_file = '../adult.filteredTest'

# define the names for each column in the data
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income']

# import the training and test data using Pandas
train = pd.read_csv(train_file, header=None, names=column_names)
test = pd.read_csv(test_file, header=None, names=column_names)

# divide the data into X and y arrays
X_train = train.drop(columns=['income'])
y_train = train['income']

X_test = test.drop(columns=['income'])
y_test = test['income']

# synchronize the native-country category between the training and test data. Some categories appear in the training
# data that don't appear in the test data, so this will ensure both datasets have the same categories
possible_categories = [" United-States", " Cambodia", " England", " Puerto-Rico", " Canada", " Germany", " Outlying-US(Guam-USVI-etc)", " India", " Japan", " Greece", " South", " China", " Cuba", " Iran", " Honduras", " Philippines", " Italy", " Poland", " Jamaica", " Vietnam", " Mexico", " Portugal", " Ireland", " France", " Dominican-Republic", " Laos", " Ecuador", " Taiwan", " Haiti", " Columbia", " Hungary", " Guatemala", " Nicaragua", " Scotland", " Thailand", " Yugoslavia", " El-Salvador", " Trinadad&Tobago", " Peru", " Hong", " Holand-Netherlands"]
dtype = pd.CategoricalDtype(categories=possible_categories)
X_train["native-country"] = pd.Series(X_train["native-country"], dtype=dtype)
X_test["native-country"] = pd.Series(X_test["native-country"], dtype=dtype)

# in order to make Naive-Bayes work with text data, we have to do some preprocessing. This uses Pandas' get_dummies
# method to enable one-hot encoding of the test data. One-hot encoding represents categorical data as binary vectors,
# where each vector element corresponds to a unique category and only one element is set to 1
text_values = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for val in text_values:
    X_train = pd.get_dummies(X_train, prefix=[val], columns=[val], drop_first=True)
    X_test = pd.get_dummies(X_test, prefix=[val], columns=[val], drop_first=True)

# convert the 'y' data to binary values (<=50K = 0; >50K = 1)
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)


# Gaussian Naive-Bayes
gaussian = GaussianNB()
gaussian = gaussian.fit(X_train, y_train)

# make predictions about the test data
y_pred = gaussian.predict(X_test)

# test and report the accuracy of the predictions
accuracyGauss = accuracy_score(y_test, y_pred) * 100
print("Prediction Accuracy:")
print(f'Gaussian NB: {accuracyGauss:.2f}%')

# train the model using Multinomial Naive-Bayes
multinomial = MultinomialNB()
multinomial = multinomial.fit(X_train, y_train)

# make predictions about the test data
y_pred = multinomial.predict(X_test)

# test and report the accuracy of the predictions
accuracyMult = accuracy_score(y_test, y_pred) * 100
print(f'Multinomial NB: {accuracyMult:.2f}%')

###################################################

# Bernoulli Naive-Bayes
bernoulli = BernoulliNB()
bernoulli = bernoulli.fit(X_train, y_train)

# make predictions about the test data
y_pred = bernoulli.predict(X_test)

# test and report the accuracy of the predictions
accuracyBernoulli = accuracy_score(y_test, y_pred) * 100
print(f'Bernoulli NB: {accuracyBernoulli:.2f}%')

##################################################

# Complement Naive-Bayes
complement = ComplementNB()
complement = complement.fit(X_train, y_train)

# make predictions about the test data
y_pred = complement.predict(X_test)

# test and report the accuracy of the predictions
accuracyComp = accuracy_score(y_test, y_pred) * 100
print(f'Complement NB: {accuracyComp:.2f}%')
