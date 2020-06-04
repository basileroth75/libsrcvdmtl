import pandas as pd
from sklearn.metrics import mean_squared_error
from libsrcvdmtl.Model import Model


class PyModel(Model):
    """
    Generic class that wraps around a sklearn compatible estimator, e.g. XGBoost or sklearn models. Being a subclass of
    sklearn's BaseEstimator allows to use RandomSearchCV or GridSearchCV on various models (self.estimator). We also use
    this class to store the order of the features in our data set (useful for XGBoost) and which columns (features) we
    discard from the data set.
    """

    def __init__(self, estimator=None, drop_columns=list(), model_path="pym-generic.pkl"):
        """
        Constructor
        :param estimator: (sklearn compatible object) estimator that has the fit, predict and score functions.
        :param drop_columns: (list) list of strings containing the names of the features to drop from the data sets
                                    X_train and X_test
        :param model_path: (string) path under which the model will be saved if self.save_model is called.
        """
        # Apparently all input args must have default values according to sklearn's convention
        super(PyModel, self).__init__(model_path)
        self.estimator = estimator
        self.drop_columns = drop_columns
        # Initialize self.features to None, will update when fit is called
        self.features = None

    def fit(self, X_train, y_train, **kwargs):
        """
        Calls self.estimator's fit function on X_train and y_train
        :param X_train: (Pandas DataFrame) training data set
        :param y_train: (Pandas Series) targets for the corresponding entries in X_train
        :param kwargs: extra keyword arguments that will be passed to self.estimator's fit function
        :return: None
        """
        # Dropping the features that are in self.drop_columns
        X_train_drop = X_train.drop(self.drop_columns, axis=1)
        # Extracting features order
        self.features = X_train_drop.columns
        self.estimator.fit(X_train_drop.values, y_train.values, **kwargs)
        return self  # apparently self must be returned for compatibility reasons in sklearn

    def predict(self, X_test, **kwargs):
        """
        Call self.estimator's predict function on X_test
        :param X_test: (Pandas DataFrame) data frame on which to predict
        :param kwargs: extra keyword arguments that will be passed to self.estimator's predict function
        :return: (Pandas DataFrame) predicitions for each entry in X_test
        """
        # Can only apply fit if model has been fit before
        assert self.features is not None
        # Dropping the features that are in self.drop_columns
        X_test_drop = X_test.drop(self.drop_columns, axis=1)[self.features]
        return pd.Series(self.estimator.predict(X_test_drop.values, **kwargs))

    def score(self, X_test, y_test, **kwargs):
        """
        Scores the prediction using sklearn's MSE function
        :param X_test: (Pandas DataFrame) data frame on which predict will be called
        :param y_test: (Pandas Series) data frame that contains the true values that should have been predicted
        :return: (float) MSE score for our prediction on X_test compared to y_test
        """
        # GridSearchCV and RandomSearchCV MAXIMIZE the score!!!
        return -mean_squared_error(y_test, self.predict(X_test))
