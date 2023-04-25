from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import loguniform
from sklearn import set_config; set_config(display='diagram')
import random

# split the data
def split_data(X, y, split=0.3):
    """ This function splits the data into training and test set. 
    By default it splits the data in 70/30% """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split, 
        random_state=42)
    return X_train, X_test, y_train, y_test