from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.preprocessing import LabelBinarizer, KBinsDiscretizer
from sklearn.metrics import accuracy_score
import pandas as pd

outfile = "NB_output.txt"

f = open(outfile, "w")
f.write("Naive-Bayes\n\n")


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
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for val in categorical_features:
    X_train = pd.get_dummies(X_train, prefix=[val], columns=[val], drop_first=True)
    X_test = pd.get_dummies(X_test, prefix=[val], columns=[val], drop_first=True)

# convert the 'y' data to binary values (<=50K = 0; >50K = 1)
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print("Prediction Accuracy:")
f.write("Prediction Accuracy:\n")
###################################################

# Gaussian Naive-Bayes

# train the model
gaussian = GaussianNB()
gaussian = gaussian.fit(X_train, y_train)

# make predictions about the test data
y_pred = gaussian.predict(X_test)

# test and report the accuracy of the predictions
accuracyGauss = accuracy_score(y_test, y_pred) * 100
print(f'Gaussian NB: {accuracyGauss:.2f}%')
f.write(f'Gaussian NB: {accuracyGauss:.2f}%\n')

##################################################

# Multinomial NB

# train the model
multinomial = MultinomialNB()
multinomial = multinomial.fit(X_train, y_train)

# make predictions about the test data
y_pred = multinomial.predict(X_test)

# test and report the accuracy of the predictions
accuracyMult = accuracy_score(y_test, y_pred) * 100
print(f'Multinomial NB: {accuracyMult:.2f}%')
f.write(f'Multinomial NB: {accuracyMult:.2f}%\n')

##################################################

# Complement Naive-Bayes

# train the model
complement = ComplementNB()
complement = complement.fit(X_train, y_train)

# make predictions about the test data
y_pred = complement.predict(X_test)

# test and report the accuracy of the predictions
accuracyComp = accuracy_score(y_test, y_pred) * 100
print(f'Complement NB: {accuracyComp:.2f}%')
f.write(f'Complement NB: {accuracyComp:.2f}%\n')

###################################################

# Bernoulli Naive-Bayes

# Bernoulli NB expects binary features, so some additional preprocessing is needed. The one-hot encoding for categorical
# values is good enough, but we need to discretize the continuous numeric values. This is done using KBinsDiscretizer
# using the k-means strategy, which uses k-means to cluster the data into discrete bins. This method, along with the
# number of bins, was selected using trial and error

# discretize the continuous features
continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')

X_train[continuous_features] = discretizer.fit_transform(X_train[continuous_features])
X_test[continuous_features] = discretizer.transform(X_test[continuous_features])

# train the model using Bernoulli NB
bernoulli = BernoulliNB()
bernoulli = bernoulli.fit(X_train, y_train)

# make predictions about the test data
y_pred = bernoulli.predict(X_test)

# test and report the accuracy of the predictions
accuracyBernoulli = accuracy_score(y_test, y_pred) * 100
print(f'Bernoulli NB: {accuracyBernoulli:.2f}%')
f.write(f'Bernoulli NB: {accuracyBernoulli:.2f}%\n')

f.close()