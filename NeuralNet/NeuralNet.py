import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
filename = "../adult.filtered"
filenameTest = '../adult.filteredTest'
# cols
columnNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
# Read in the training data
train_data = pd.read_csv(filename, header=None, names=columnNames)
test_data = pd.read_csv(filenameTest, header=None, names=columnNames)
# Merge the data together
data = pd.concat([train_data, test_data], axis=0)

# Preprocessing the data
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 2, 4, 10, 11, 12]),
        ('cat', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])
    ])
x = ct.fit_transform(x)

train_data = data[:len(train_data)]
test_data = data[len(train_data):]

# Spliting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Building neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Converting the end result to either over 50 k or under 50 k
y_train = np.where(y_train == ' >50K', 1, 0)
y_test = np.where(y_test == ' >50K', 1, 0)
# Traing the model
final = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))