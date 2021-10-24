# Import packages to be used in the program.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Read the CSV file, save it as a variable called "data" and saves the values in the Temp and Yield columns as lists of
# X and Y values.

data = pd.read_csv('yield.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Utilise the train_test_split function to split the data into data for training and data for testing. The splitting of
# the data is done completely randomly. The size of the test set is set at 40% of the size of the original dataset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Use the linear regression function from Scikit-learn to fit the linear regressor (line of best fit) to the training
# dataset. The numpy linspace function is used to specify the size of the x array - min (0) and max (120) values,
# and the maximum number of values that can be in the set (100). The predict function is used to predict the values
# in the y_array. The x and y arrays are then plotted.

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
x_array = np.linspace(0, 120, 100)
y_array = linear_regressor.predict(x_array.reshape(x_array.shape[0], 1))
plt.plot(x_array, y_array)

# Set the degree of the Polynomial Regression to 2 and convert the x variable test and train data into a quadratic data
# matrix.

poly_features = PolynomialFeatures(degree=2)
X_train_quadratic = poly_features.fit_transform(X_train)
X_test_quadratic = poly_features.transform(X_test)

# Fit the linear regressor/line of best fit to the quadratic x training data and the y training data.
# Convert the linear x_array to quadratic.

poly_regression = LinearRegression()
poly_regression.fit(X_train_quadratic, y_train)
x_array_quadratic = poly_features.transform(x_array.reshape(x_array.shape[0], 1))

# Plot the linear x_array data and the quadratic y data in a scatterplot.

plt.plot(x_array, poly_regression.predict(x_array_quadratic), c='r', linestyle='--')
plt.title('Yield of scientific experiment at 5 different temperature levels')
plt.xlabel('Temperature')
plt.ylabel('Yield')
plt.axis([40, 110, 1, 4])
plt.grid(True)
plt.scatter(X_train, y_train, c='black')
plt.show()
