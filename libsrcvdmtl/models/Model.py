import dill as pickle
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class Model(BaseEstimator, RegressorMixin):
    """
    Generic class that represents various Models and follows the sklearn API. Subclasses have to define init, fit,
    predict and score functions but this class provides implementations of the generic methods save_model and
    load_model.
    """

    # ---
    # Class methods
    # ---

    @staticmethod
    def load_model(model_path):
        """
        Loads a model
        :param model_path: (string) path to the model to load
        :return: (Model) an object of a subclass of Model
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    # ---
    # Public methods
    # ---

    def __init__(self, model_path):
        """
        Constructor
        :param model_path: (string) Full path under which the model will be saved if self.save_model is called.
        """
        self.model_path = model_path

    # def fit(self, X_train, y_train):
    #     pass
    #
    # def predict(self, X_test):
    #     pass
    #
    # def score(self, X_test, y_test):
    #     pass

    def save_model(self):
        """
        Saves a model (self) as a .pkl file under the models/ folder. The model will be saved under the "model_path"
        path.
        :return: None
        """
        with open(self.model_path, "wb") as file:
            pickle.dump(self, file)
        return
