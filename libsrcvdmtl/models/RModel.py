import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from src.models.Model import Model

pandas2ri.activate()


class RModel(Model):
    def __init__(self, dataset):
        """
        Constructor
        """
        super(RModel, self).__init__(dataset)

    def fit(self, X_train, y_train):
        """
        Wrapper that calls train.R
        :param X_train:
        :param y_train:
        :return:
        """
        r_source = robjects.r['source']
        r_source('./model-R/train.R')
        r_func = robjects.globalenv['train']
        r_func()
        return

    def predict(self, X_test):
        """
        Wrapper that calls prediction.R
        :param X_test:
        :return:
        """
        r_source = robjects.r['source']
        r_source('./model-R/prediction.R')
        r_func = robjects.globalenv['prediction']
        r_df = r_func(self.dataset.shape_path, self.dataset.shape_key)
        pred = pandas2ri.ri2py(r_df)
        pred.columns = ['shape_ID', 'prediction']
        pred.set_index('shape_ID', inplace=True)
        return pred

    def score(self, X_test, y_test):
        pass
        # return mean_squared_error(y_test, self.predict(X_test).prediction)


if __name__ == "__main__":
    from src.data.DataSet import DataSet
    import os

    shape_path = "../data/processed/Secteur_casernes_operationnels/Secteur_de_casernes_opérationnels.shp"
    shape_key = "NO_CAS_OP"
    delta_t = 15

    data = DataSet(shape_path, shape_key, delta_t)

    rm = RModel(dataset=data)

    print(os.getcwd())
    # rm.fit(data.X_train, data.y_train)
    rm.predict(data.X_test)
    print(rm.score(data.X_test, data.y_test))
