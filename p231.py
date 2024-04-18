import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt(r"books.csv", error_bad_lines = False)
x = dataset.iloc[:, [4, 11]].values
y = dataset.iloc[:, 3].values

model = Sequential()
model.add(Dense(115, input_dim = 8, activation = 'relu'))
model.add(Dense(117, activation = 'relu'))
model.add(Dense(53, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()