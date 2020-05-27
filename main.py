import pickle

import pandas as pd
import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


dataset = pd.read_csv('new_new.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression().fit(X_train, y_train)

result = regressor.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * result))
#
filename = 'LinearRegression.pk1'
# pickle.dump(regressor, open(filename, 'wb'))


regressor = pickle.load(open(filename, 'rb'))
# Predicting the Test set results
#a = [[140,4679511,315005,3330,600.3380000000002,10.387,40.412,0.4270000000000001]]
y_pred = regressor.predict(X_test)
# print(y_pred[0])
print()

r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("R-Squared - "+str(r2))
print()

r2 = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error - "+str(r2))
print()

r2 = mean_squared_error(y_test, y_pred)
print("Mean Squared Error - "+str(r2))
print()