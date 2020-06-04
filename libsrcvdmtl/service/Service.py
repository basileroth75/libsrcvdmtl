from src.models.Model import Model
from src.data.DataSet import DataSet
import pandas as pd


class FirstResponderService(object):
    """
    Class that should be instantiated when the Flask service is launched. It contains the files that are required in
    order to make a prediction
    """

    def __init__(self, end_date_path, model_path):
        """
        Constructor
        :param end_date_path: (string) path to the end_date.parquet file that is generated by
                                       airflow/01_Create_Train_Set.ipynb
        :param model_path: (string) path to the pickle model that the user wishes to call. Typically should be generated
                                    using the notebook airflow/02_Train_Models.ipynb on the output of
                                    airflow/01_Create_Train_Set.ipynb.
        """
        self.end_date_path = end_date_path
        # Load the model using Model's generic load_model function
        self.model = Model.load_model(model_path)
        # Store the files that Flask has to monitor
        self.dependency_files = [end_date_path, model_path]

    def predict(self):
        """
        Wrapper function around self.model.predict that is used in Flask
        :return:
        """
        # We have to create a X_test data frame that is compatible with our self.model object
        X_test = self.transform()
        # Return the prediction of our model
        return DataSet.undiff(X_test, self.model.predict(X_test))

    def transform(self):
        """
        :return:
        """

        # Read last available date of shape counts and statistics
        df = pd.read_parquet(self.end_date_path)

        return df


if __name__ == "__main__":
    end_date_path = "../../data/datasets/end_date.parquet"
    model_path = "../../data/models/xgb-model.pkl"
    service = FirstResponderService(end_date_path, model_path)

    service.predict(pd.to_datetime("2017-01-01"), 20, 10, 0.5, 100)