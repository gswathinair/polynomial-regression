import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

house_data = pd.read_csv('./data/house_rental_data.csv.txt', index_col='Unnamed: 0')

# Defining features and target
X = house_data[house_data.columns[0:5]]
Y = house_data[house_data.columns[-1]]

print("Y", Y.shape)
print("X", X.shape)

# Divide the data into testing and training.
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.33, random_state=42)

print(trainX.shape, testX.shape, trainY.shape, testY.shape)

model = []

# We consider upto 5 degrees of polynomial equation.
DEGREE_OF_EQUATION = 5

for deg in range(1, DEGREE_OF_EQUATION):
    """
    PolynomialFeatures():
    Generate a new feature matrix consisting of all polynomial combinations of the features 
    with degree less than or equal to the specified degree. For example, if an input sample 
    is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    """

    pol = PolynomialFeatures(degree=deg)

    # polynomial fit_transform training data
    data_tf = pol.fit_transform(trainX)

    # Define and fit LinearRegression model
    lr = LinearRegression()
    lr.fit(data_tf, trainY)

    # polynomial fit_transform testing data
    testX_tf = pol.transform(testX)

    # Predict training and testing results
    y_pred = lr.predict(testX_tf)
    y_pred_train = lr.predict(data_tf)

    # Calculate mean_absolute_error for training and testing
    err_test = mean_absolute_error(y_pred, testY)
    err_train = mean_absolute_error(y_pred_train, trainY)

    # As degree increases, model may overfit.
    # Decide the best polynomial degree based on the least values of mean_absolute_error for TESTING DATA.
    print(deg, np.round(err_train), np.round(err_test))
