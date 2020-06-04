import numpy as np
from sklearn.metrics import mean_squared_error
from src.models.Model import Model


class BaselineModel(Model):
    """
    Class representing our baseline for the forecasting of emergency calls in Montreal
    """

    def __init__(self, model_path="bl-generic.pkl"):
        """
        Constructor
        :param model_path: (string) path where the model will be saved
        """
        super(BaselineModel, self).__init__(model_path)

    def fit(self, X_train, _):
        """
        Fits the model, i.e. extracts the most recent value of "shp_cumul_mean". This method requires the columns
        shape_key and shp_cumul_mean to work.
        :param X_train: (Pandas DataFrame) training data set
        :param _: (dummy) this function doesn't need y_train
        :return: None
        """
        assert (np.any(X_train.columns == "shp_cumul_mean"))
        assert (np.any(X_train.columns == "shape_key"))
        # According to sklearn's convention, attributes defined in fit should have an underscore (_) as a suffix
        self.pred_ = X_train.groupby("shape_key").shp_cumul_mean.last().to_frame("bl_pred").reset_index()
        return

    def predict(self, X_test):
        """
        Predicts on X_test once the model has been fit. This method requires the columns call_date_time and shape_key to
        work.
        :param X_test: (Pandas DataFrame) data frame on which to predict
        :return: (Pandas DataFrame) predicitions for each entry in X_test
        """
        # Must call fit first
        try:
            getattr(self, "pred_")
        except AttributeError:
            raise RuntimeError("You must call self.fit() first")

        assert (np.any(X_test.columns == "call_date_time"))
        assert (np.any(X_test.columns == "shape_key"))
        # Outputting the same prediction for all call date times
        return X_test[["shape_key"]].merge(self.pred_, on="shape_key").bl_pred

    def score(self, X_test, y_test, **kwargs):
        """
        Scores the prediction using sklearn's MSE function
        :param X_test: (Pandas DataFrame) data frame on which predict will be called
        :param y_test: (Pandas Series) data frame that contains the true values that should have been predicted
        :return: (float) MSE score for our prediction on X_test compared to y_test
        """
        # We use the neg MSE to be consistant
        return -mean_squared_error(y_test, self.predict(X_test))


if __name__ == "__main__":
    from src.data.DataSet import DataSet

    shape_path = "../../data/processed/Sect_cas_op/Sect_cas_op.shp"
    shape_key = "NO_CAS_OP"
    delta_tp = 15
    delta_tu = 30

    data = DataSet(shape_path, shape_key, delta_tp, delta_tu)

    bm = BaselineModel("../../models/bl-generic.pkl")
    bm.fit(data.X_train, data.y_train)
    pred = bm.predict(data.X_test)
    print(pred)
    print(bm.score(data.X_test, data.y_test))
