from multiprocessing import Pool, cpu_count
import pandas as pd
import pyflux as pf
from sklearn.metrics import mean_squared_error
from libsrcvdmtl.models.Model import Model


class ArimaModel(Model):
    """
    ArimaModel

    Provides a forecast using an Arima model from the PyFlux library
    """

    def __init__(self, ar=1, ma=3, diff=0, model_path="ari-generic.pkl"):
        """
        Constructor
        :param model_path (string)
        :param ar: (int)
        :param ma: (int)
        :param diff: (int)
        """
        super(ArimaModel, self).__init__(model_path)
        self.ar = ar
        self.ma = ma
        self.diff = diff
        self.estimators = {}
        self.estimators_params = {}

    def _group_fit(self, group):
        # Get shape code (dict key)
        n_shape = group["shape_key"].iloc[0]
        # Create Arima model
        estimator = pf.ARIMA(data=group, ar=self.ar, ma=self.ma, integ=self.diff, target="pred", family=pf.Normal())
        estimator.fit()
        return pd.DataFrame(index=[n_shape], data=[estimator])

    def fit(self, X_train, y_train):
        # PyFlux models need to have target in same dataframe
        merged = X_train[["shape_key", "call_date_time"]].join(y_train.to_frame("pred"), how="left"). \
            groupby("shape_key")

        # Call _group_fit in parallel!
        with Pool(cpu_count()) as p:
            ret_list = p.map(self._group_fit, [group for name, group in merged])
        p.terminate()
        self.estimators = pd.concat(ret_list).to_dict()[0]

    def _group_predict(self, group):
        # Get shape code (dict key)
        n_shape = group["shape_key"].iloc[0]
        pred = self.estimators[n_shape].predict(h=group.shape[0])
        return pred

    def predict(self, X_test, **kwargs):
        # Check if estimator dict is not empty
        assert self.estimators
        grouped = X_test[["call_date_time", "shape_key"]].groupby("shape_key")

        with Pool(cpu_count()) as p:
            ret_list = p.map(self._group_predict, [group for name, group in grouped])
            p.terminate()
        return pd.concat(ret_list).reset_index(drop=True).pred

    def score(self, X_test, y_test, **kwargs):
        return -mean_squared_error(y_test, self.predict(X_test))


if __name__ == "__main__":
    from src.data.DataSet import DataSet
    import matplotlib.pyplot as plt

    shape_path = "../../data/processed/Sect_cas_op/Sect_cas_op.shp"
    shape_key = "NO_CAS_OP"
    delta_tp = 15
    delta_tu = 30

    data = DataSet(shape_path=shape_path, shape_key=shape_key, delta_tp=delta_tp, delta_tu=delta_tu)

    am = ArimaModel(1, 3, 0)
    am.fit(data.X_train, data.y_train)

    print("\nfit done\n")

    pred = am.predict(data.X_test)

    print("mse = %.2f\n" % mean_squared_error(data.y_test, pred))

    pred.plot(color='b', marker='.', alpha=0.8)
    data.y_test.plot(color='r', marker='.', alpha=0.8)
    plt.show()
