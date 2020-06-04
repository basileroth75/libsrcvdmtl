import sqlalchemy
import pandas as pd
import math
import numpy as np
import glob
import os

from multiprocessing import Pool, cpu_count
from src.data.meteo_utils import get_meteo


def day_cosinus(x):
    return np.cos((2 * math.pi / 365) * (x - 1))


def day_sinus(x):
    return np.sin((2 * math.pi / 365) * (x - 1))


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    Returns the distance (in km)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


class DataSetDensity(object):
    # ---
    # Class methods
    # ---

    # Shared by all instances
    __interim_parquet_path = "../../data/processed/IPR.parquet"

    @staticmethod
    def update_dataset():
        """
        Updates the local data set from the SQL database
        :return:
        """

        # Cleaning directory first, remove old parquet files since database was updated
        for file in glob.glob("../../data/density_datasets/*.parquet"):
            os.remove(file)

        # Query from SQL database
        engine = sqlalchemy.create_engine('postgresql://dsuser:datascience17@10.1.35.45:5432/postgres')
        # Look up table DB_INCIDENT.
        df = pd.read_sql_query('SELECT * FROM ssdo_owner.db_incident', engine)
        # Changes occured in 2012, so previous data should be discarded
        df = df.loc[(df['version_log_no'] >= 9) & (df['call_date_time'].dt.year >= 2012)]
        # Focus on specific first-responders interventions
        df = df[df['incident_type'].str.startswith('1')]

        # Filter needed columns
        selected_coumns = ["call_date_time", "latitude", "longitude"]

        # Save *intermediate* training and test sets to .parquet
        df[selected_coumns].reset_index(drop=True).to_parquet(DataSetDensity.__interim_parquet_path)

        return

    @staticmethod
    def load_dataset(path):
        """
        Utility function that loads the .parquet file located at path and returns feature and target dataframes.
        Expects the target column to be named "count_pred"
        :param path:
        :return: X: (Pandas DataFrame) features dataframe
                 Y: (Pandas DataFrame) target dataframe
        """
        df = pd.read_parquet(path)
        assert (np.any(df.columns == "count_pred"))
        X = df.drop("count_pred", axis=1)  # Keep all features except count_pred
        y = df.count_pred
        return X, y

    @staticmethod
    def _get_meteo(db_begin_date, db_end_date):
        """

        :return:
        """

        df_meteo = get_meteo(db_begin_date.year, db_end_date.year)

        # Remove useless columns.
        # Wind Chill and Hmdx could be removed as they are highly correlated with Temp. Furthermore, Dew Point Temp is
        # also highly correlated with Temp.
        df_meteo = df_meteo.drop(["Year", "Month", "Day", "Time", "Data Quality", "Temp Flag", "Weather",
                                  "Dew Point Temp Flag", "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag",
                                  "Stn Press Flag", "Hmdx Flag", "Wind Chill Flag", "Visibility Flag"], axis=1)

        # Convert column to datetime object
        df_meteo['Date/Time'] = pd.to_datetime(df_meteo['Date/Time'])

        # Climate data must match the entries in our data frame
        df_meteo = df_meteo.loc[(df_meteo['Date/Time'] <= db_end_date) &
                                (df_meteo['Date/Time'] >= db_begin_date), :]

        # Parse ugly column names, remove parentheses and convert spaces to underscores
        df_meteo.columns = df_meteo.columns.str.replace(r" \(.*\)", "").str.replace(r" ", "_")

        # Fix broken dtypes
        df_meteo[["Temp", "Dew_Point_Temp", "Visibility", "Stn_Press"]] = \
            df_meteo[["Temp", "Dew_Point_Temp", "Visibility", "Stn_Press"]].astype(float)

        # Climate data is by hours, we resample by day
        df_meteo = df_meteo.resample("1D", on="Date/Time").mean().reset_index()

        return df_meteo

    # ---
    # Public methods
    # ---

    def __init__(self, rayon, delta_tp=15, delta_tu=30):
        """
        Constructor
        :param rayon: size of the radius of nb calls to consider
        :param delta_tp: timeframe to predict
        :param delta_tu: testing window
        """
        # For now, assert that update time is greater than prediction time (we use delta_tu to create the test set)
        assert (delta_tu >= delta_tp)
        self.rayon = rayon
        self.delta_tp = delta_tp
        self.delta_tu = delta_tu

        self.today = pd.to_datetime('today')
        self.db_end_date = None
        self.db_begin_date = None
        self.split_train_test_date = None

        self.df_incident = None
        # Check if a data set that matches rayon and delta_t exists
        exists, self.train_parquet_path, self.test_parquet_path = self._density_dataset_exists()
        if not exists:
            # If no such parquet files exist, call preprocess to create them
            self._preprocess(self.train_parquet_path, self.test_parquet_path)

        self.X_train, self.y_train = DataSetDensity.load_dataset(self.train_parquet_path)
        self.X_test, self.y_test = DataSetDensity.load_dataset(self.test_parquet_path)

    def update_dataset(self):
        DataSetDensity.create_interim_dataset()
        self.__init__(self.rayon, self.delta_tp, self.delta_tu)
        return self

    def tscv(self, n_splits=10):
        """
        Cross-validation for time series.
        Creates n_splits validation sets of self.delta_tu days. After each pass, the validation data is added to the
        training set. The last self.delta_tp days of each training sets are removed to avoid having information of the
        validation set.
        :param n_splits: (int) number of CV splits
        :return: (list) list of tuples containing the indices of the training and test set at each iteration
        """

        # call_date_time must be a column
        assert (np.any(self.X_train.columns == "call_date_time"))
        cdt = self.X_train.call_date_time
        # Total number of days available to train
        end_time = cdt.max()
        n_days_train = (end_time - cdt.min()).days

        # Get maximal number of splits
        max_splits = int(n_days_train / self.delta_tu)
        # This leaves at least 1 split to train on
        assert (n_splits < max_splits)

        splits = []
        for split in np.arange(n_splits, 0, -1):
            train = np.where(cdt < end_time - pd.Timedelta(split * self.delta_tu + self.delta_tp - 1, "d"))
            test = np.where((cdt >= end_time - pd.Timedelta(split * self.delta_tu, "d")) &
                            (cdt < end_time - pd.Timedelta((split - 1) * self.delta_tu, "d")))

            splits.append((train[0], test[0]))

        return splits

    # ---------------
    # Private methods
    # ---------------

    def _density_dataset_exists(self):
        """
        :return:
        """
        try:
            dataset_exists = pd.read_csv("../../data/density_datasets/density_dataset_exists.csv")
        except IOError:
            dataset_exists = pd.DataFrame(columns=["delta_t", "rayon", "dataset_train_path", "dataset_test_path"])

        dataset = dataset_exists[(dataset_exists["rayon"] == self.rayon)]

        if len(dataset) > 0:
            # Should exist
            try:
                # Check to be sure
                pd.read_parquet(dataset.dataset_train_path.iloc[0])
                # No need to call preprocess
                return True, dataset.dataset_train_path.iloc[0], dataset.dataset_test_path.iloc[0]
            except IOError:
                # If it existed in the csv but the parquet is missing, call preprocess
                return False, dataset.dataset_train_path.iloc[0], dataset.dataset_test_path.iloc[0]
        else:
            train_parquet_path = "../../data/density_datasets/dataset_" + str(self.rayon) + "km" + "_train.parquet"
            test_parquet_path = "../../data/density_datasets/dataset_" + str(self.rayon) + "km" + "_test.parquet"

            dataset_exists.append(pd.DataFrame([
                [self.rayon, self.delta_tp, self.delta_tu, train_parquet_path, test_parquet_path]],
                columns=["rayon", "delta_t", "delta_tu", "dataset_train_path", "dataset_test_path"])). \
                to_csv("../../data/density_datasets/density_dataset_exists.csv", index=False, encoding="utf-8")

            return False, train_parquet_path, test_parquet_path

    def _preprocess(self, train_save_path, test_save_path):
        # Read local data set, to update, call update_interim_dataset
        try:
            df = pd.read_parquet(DataSetDensity.__interim_parquet_path)
        except IOError:
            print("%s not found, running DataSet.create_interim_dataset()\n" % DataSetDensity.__interim_parquet_path)
            DataSetDensity.create_interim_dataset()
            df = pd.read_parquet(DataSetDensity.__interim_parquet_path)

        df = df.sort_values("call_date_time")

        self.df_incident = df

        df["year"] = df.call_date_time.dt.year
        df["month"] = df.call_date_time.dt.month
        df["day"] = df.call_date_time.dt.day
        df["dayofyear"] = df.call_date_time.dt.dayofyear

        df["dayofyear_cosinus"] = day_cosinus(df["dayofyear"])
        df["dayofyear_sinus"] = day_sinus(df["dayofyear"])

        timedelta_tp = pd.Timedelta(self.delta_tp, "d")
        df_filtered = df.loc[df.call_date_time >= (df.call_date_time.min() + timedelta_tp), :].copy()

        df_filtered["group"] = np.mod(df_filtered.index.values, cpu_count())
        grouped = df_filtered.groupby("group")

        with Pool(cpu_count()) as p:
            ret_list = p.map(self._group_predict, [group for name, group in grouped])
        p.terminate()

        df_calls = pd.DataFrame([item for sublist in ret_list for item in sublist],
                                columns=["call_date_time", "count_pred"])

        df_filtered = df_filtered.merge(df_calls, on="call_date_time")

        df_feature = pd.read_parquet("../../data/processed/usines.parquet")
        df_feature = df_feature.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        df_filtered["count_facto"] = df_filtered.apply(lambda row: self._spatial_window(df_feature, row.latitude,
                                                                                        row.longitude), axis=1)
        df_feature = pd.read_parquet("../../data/processed/ecoles.parquet")
        df_feature = df_feature.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        df_filtered["count_schools"] = df_filtered.apply(lambda row: self._spatial_window(df_feature, row.latitude,
                                                                                          row.longitude), axis=1)
        df_feature = pd.read_parquet("../../data/processed/residences_personnes_agees.parquet")
        df_feature = df_feature.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        df_filtered["count_resid"] = df_filtered.apply(lambda row: self._spatial_window(df_feature, row.latitude,
                                                                                        row.longitude), axis=1)

        # First and last dates
        self.db_end_date = df_filtered.call_date_time.max()
        self.db_begin_date = df_filtered.call_date_time.min()

        # Entries with time lag have NaN values which correspond to incomplete/false entries, we must remove them
        df_filtered = df_filtered.dropna(axis=0, how="any")

        df_filtered.call_date_time = pd.to_datetime(df_filtered.call_date_time.dt.date)

        # Get climate data and join data frames
        df_filtered = df_filtered.merge(DataSetDensity._get_meteo(self.db_begin_date, self.db_end_date), 
                                        left_on="call_date_time", right_on="Date/Time",
                                        how="left").drop("Date/Time", axis=1)

        # Split in test and train sets, test on delta_tu (>= delta_tp)
        self.split_train_test_date = self.db_end_date - pd.Timedelta(self.delta_tu - 1, unit="d")
        test = df_filtered.loc[df_filtered.call_date_time >= self.split_train_test_date, :]
        train = df_filtered.loc[df_filtered.call_date_time < self.split_train_test_date, :]

        # The last "delta_t" days have "incomplete" predictions, remove those rows from train set
        # Note that the last rows of the test set were already removed by the previous dropna()
        end_date = train.call_date_time.max() - pd.Timedelta(self.delta_tp, "d")
        train = train.loc[train.call_date_time <= end_date, :]

        # Save to .parquet
        train.reset_index(drop=True).to_parquet(train_save_path)
        test.reset_index(drop=True).to_parquet(test_save_path)

        return

    def _calls_space_time(self, tp, lat, long):
        df_temp = self.df_incident.loc[(self.df_incident.call_date_time <= tp) &
                                       (self.df_incident.call_date_time > tp - pd.Timedelta(self.delta_tp, "d")), :]
        return (haversine(df_temp["longitude"], df_temp["latitude"], long, lat) < self.rayon).sum()

    def _spatial_window(self, df_feat, lat, long):
        return (haversine(df_feat["longitude"], df_feat["latitude"], long, lat) < self.rayon).sum()

    def _group_predict(self, df):
        nb_calls = []
        for index, row in df.iterrows():
            nb_calls.append([row.call_date_time,
                             self._calls_space_time(row.call_date_time, row.latitude, row.longitude)])
        return nb_calls


if __name__ == "__main__":
    # Create interim parquet file that is used by the preprocess method
    # DataSet.create_interim_dataset()
    rayon = 1
    delta_tp = 15
    delta_tu = 30

    data = DataSetDensity(rayon, delta_tp=delta_tp, delta_tu=delta_tu)
    splits = data.tscv(1)

    cdt = data.X_train.call_date_time

    for split in splits:
        train_beg = cdt.iloc[split[0][0]]
        train_end = cdt.iloc[split[0][-1]]
        print("train beginning = %s\ntrain end = %s\n" % (train_beg, train_end))
        test_beg = cdt.iloc[split[1][0]]
        test_end = cdt.iloc[split[1][-1]]
        print("test beginning = %s\ntest end = %s\n" % (test_beg, test_end))
