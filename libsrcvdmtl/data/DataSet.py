import glob
import os
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import holidays
import numpy as np
import pandas as pd
import sqlalchemy
# from pyflux import ARIMA, Normal  # discarded feature
import tsfresh
from shapely.geometry import Point

from src.data.meteo_utils import get_meteo


class DataSet(object):
    """

    """

    # ---
    # Class methods
    # ---

    # Shared by all instances
    _interim_parquet_path = "../../data/processed/IPR.parquet"
    _save_dataset_path = "../../data/datasets/"
    _sql_engine = "postgresql://dsuser:datascience17@10.1.35.45:5432/postgres"
    _sql_query = "SELECT * FROM ssdo_owner.db_incident"
    _census_dataset_path = "../../data/processed/data_census_2016.parquet"
    _fsa_shape_path = "../../data/processed/Montreal_fsa/Montreal_fsa.shp"
    _fsa_shape_key = "CFSAUID"
    _311_url = "http://donnees.ville.montreal.qc.ca/dataset/5866f832-676d-4b07-be6a-e99c21eb17e4/resource/" + \
               "2cfa0e06-9be4-49a6-b7f1-ee9f2363a872/download/requetes311.csv"

    # After exploring with notebook 14, we concluded that most features don't help much, but our prediction was slightly
    # better anyways. We keep the ones that had the highest feature importance and leave the others commented. Note that
    # the features below were already filtered by hand, that is more features could be computed (about 750).
    _tsfresh_settings = {
        # "abs_energy": None,
        # "absolute_sum_of_changes": None,
        'ar_coefficient': [{'coeff': 0, 'k': 10},
                           {'coeff': 1, 'k': 10},
                           {'coeff': 2, 'k': 10},
                           {'coeff': 3, "k": 10},
                           {'coeff': 4, 'k': 10},
                           {'coeff': 5, 'k': 10},
                           {'coeff': 6, 'k': 10},
                           {'coeff': 7, "k": 10},
                           {'coeff': 8, 'k': 10},
                           {'coeff': 9, 'k': 10},
                           {'coeff': 10, 'k': 10},
                           ],
        # "cid_ce": [{"normalize": False}],
        # "count_above_mean": None,
        # "count_below_mean": None,
        # 'cwt_coefficients': [{'coeff': 0, 'w': 5, 'widths': (5, 10, 15)},
        #                      {'coeff': 0, 'w': 10, 'widths': (5, 10, 15)},
        #                      {'coeff': 0, 'w': 15, 'widths': (5, 10, 15)},
        #                      {'coeff': 1, 'w': 5, 'widths': (5, 10, 15)},
        #                      {'coeff': 1, 'w': 10, 'widths': (5, 10, 15)},
        #                      {'coeff': 1, 'w': 15, 'widths': (5, 10, 15)},
        #                      {'coeff': 2, 'w': 5, 'widths': (5, 10, 15)},
        #                      {'coeff': 2, 'w': 10, 'widths': (5, 10, 15)},
        #                      {'coeff': 2, 'w': 15, 'widths': (5, 10, 15)}],
        # 'fft_aggregated': [{'aggtype': 'centroid'},
        #                    {'aggtype': 'variance'},
        #                    {'aggtype': 'skew'},
        #                    {'aggtype': 'kurtosis'}],
        'fft_coefficient': [{'attr': 'real', 'coeff': 0},
                            {'attr': 'real', 'coeff': 1},
                            {'attr': 'real', 'coeff': 2},
                            {'attr': 'real', 'coeff': 3},
                            {'attr': 'real', 'coeff': 4},
                            # {'attr': 'real', 'coeff': 5},
                            # {'attr': 'real', 'coeff': 6},
                            # {'attr': 'real', 'coeff': 7},
                            # {'attr': 'real', 'coeff': 8},
                            # {'attr': 'real', 'coeff': 9},
                            # {'attr': 'real', 'coeff': 10},
                            {'attr': 'imag', 'coeff': 0},
                            {'attr': 'imag', 'coeff': 1},
                            {'attr': 'imag', 'coeff': 2},
                            {'attr': 'imag', 'coeff': 3},
                            {'attr': 'imag', 'coeff': 4},
                            # {'attr': 'imag', 'coeff': 5},
                            # {'attr': 'imag', 'coeff': 6},
                            # {'attr': 'imag', 'coeff': 7},
                            # {'attr': 'imag', 'coeff': 8},
                            # {'attr': 'imag', 'coeff': 9},
                            # {'attr': 'imag', 'coeff': 10},
                            {'attr': 'abs', 'coeff': 0},
                            {'attr': 'abs', 'coeff': 1},
                            {'attr': 'abs', 'coeff': 2},
                            {'attr': 'abs', 'coeff': 3},
                            {'attr': 'abs', 'coeff': 4},
                            # {'attr': 'abs', 'coeff': 5},
                            # {'attr': 'abs', 'coeff': 6},
                            # {'attr': 'abs', 'coeff': 7},
                            # {'attr': 'abs', 'coeff': 8},
                            # {'attr': 'abs', 'coeff': 9},
                            # {'attr': 'abs', 'coeff': 10},
                            ],
        # 'friedrich_coefficients': [{'coeff': 0, 'm': 3, 'r': 30},
        #                            {'coeff': 1, 'm': 3, 'r': 30},
        #                            {'coeff': 2, 'm': 3, 'r': 30},
        #                            {'coeff': 3, 'm': 3, 'r': 30}],
        # "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
        # 'longest_strike_above_mean': None,
        # 'longest_strike_below_mean': None,
        # 'number_crossing_m': [{'m': 0}, {'m': -1}, {'m': 1}],
        # 'number_cwt_peaks': [{'n': 1}, {'n': 5}],
        # 'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 15}],
        # 'quantile': [{'q': 0.1},
        #              {'q': 0.2},
        #              {'q': 0.3},
        #              {'q': 0.4},
        #              {'q': 0.6},
        #              {'q': 0.7},
        #              {'q': 0.8},
        #              {'q': 0.9}],
        # 'range_count': [{'max': 1, 'min': -1}],
        # 'ratio_beyond_r_sigma': [{'r': 0.5},
        #                          {'r': 1},
        #                          {'r': 1.5},
        #                          {'r': 2},
        #                          {'r': 2.5}],
        # 'sample_entropy': None,
        # 'skewness': None,
        'spkt_welch_density': [{'coeff': 0},
                               {'coeff': 1},
                               {'coeff': 2},
                               {'coeff': 3},
                               {'coeff': 4},
                               # {'coeff': 5},
                               # {'coeff': 6},
                               # {'coeff': 7},
                               # {'coeff': 8},
                               # {'coeff': 9},
                               # {'coeff': 10}
                               ],
    }

    # Categories to parse the 311 data set
    _CAT_0 = ["police", "urgence", "crime", "sécurité", "surveillance", "éviction", "enquête", "intervention",
              "plainte",
              "inspection", "occupation non-autorisée", "stationnement", "alarme", "vitesse", "patrouille",
              "infraction",
              "alcool", "drogue", "violence", "trouble", "suspect", "assistance"]
    _CAT_1 = ["bruit", "événement", "activité", "culture", "sport", "loisir", "vente de garage"]
    _CAT_2 = ["rue", "voie", "circulation", "trottoir", "neige", "chaussée", "voirie", "nid-de-poule", "balai",
              "pavage",
              "public", "chantier", "déblaiement", "signalisation", "pont", "excavation", "bâtiment", "parc",
              "plomberie",
              "immeuble", "borne", "éclairage", "eau", "terrain", "puisard", "environnement", "trou d'homme", "égout",
              "tuyau", "construction", "travaux", "urbain", "réparation", "menuiserie", "édifice", "urbanisme",
              "démolition",
              "logement", "afaisement"]
    _CAT_3 = ["sale", "insalubrité", "poubelle", "matière", "collecte", "déchet", "insalubre", "animal", "mort",
              "nettoyage", "technique", "service", "graffiti", "salubrité", "bac", "sac", "encombrants", "branche",
              "arbre",
              "extermination", "gazon", "punaise", "recyclage", "élagage", "horticulture", "essouchement", "nuisance",
              "vermine", "pesticide", "écocentre", "ébulition", "résidu", "vert", "jardin", "végétaux", "animaux",
              "agrile",
              "nid"]

    @staticmethod
    def update_dataset():
        """
        Updates the local data set from the SQL database
        :return:
        """

        # Cleaning directory first, remove old parquet files since database was updated
        for file in glob.glob(DataSet._save_dataset_path + "*.parquet"):
            os.remove(file)
        # Remove the .csv too
        for file in glob.glob(DataSet._save_dataset_path + "*.csv"):
            os.remove(file)

        # Query from SQL database
        engine = sqlalchemy.create_engine(DataSet._sql_engine)
        # Look up table DB_INCIDENT.
        df = pd.read_sql_query(DataSet._sql_query, engine)
        # Changes occured in 2012, so previous data should be discarded
        df = df.loc[(df['version_log_no'] >= 9) & (df['call_date_time'].dt.year >= 2012)]
        # Focus on specific first-responders interventions
        df = df[df['incident_type'].str.startswith('1')]

        # Filter needed columns
        selected_coumns = ["call_date_time", "latitude", "longitude"]

        # Save *intermediate* training and test sets to .parquet
        df[selected_coumns].reset_index(drop=True).to_parquet(DataSet._interim_parquet_path)

        return

    @staticmethod
    def load_dataset(path):
        """
        Utility function that loads the .parquet file located at path and returns feature and target dataframes.
        Expects the target column to be named "shp_count_pred"
        :param path:
        :return: X: (Pandas DataFrame) features dataframe
                 Y: (Pandas DataFrame) target dataframe
        """
        df = pd.read_parquet(path)
        assert (np.any(df.columns == "shp_count_pred"))
        X = df.drop("shp_count_pred", axis=1)  # Keep all features except shp_count_pred
        y = df.shp_count_pred
        return X, y

    @staticmethod
    def _get_meteo(db_begin_date, db_end_date):
        """
        Function that calls src.data.meteo_utils.get_meteo, removes and parses the useless features.
        :param db_begin_date: (int) first year
        :param db_end_date: (int) last year
        :return: (Pandas DataFrame) data frame containing the filtered daily weather data for the span between the years
                                    db_begin_date and db_end_date
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

        df_meteo = df_meteo.drop(["Visibility", "Hmdx", "Wind_Chill"], axis=1)

        # Fix broken dtypes
        df_meteo[["Temp", "Dew_Point_Temp", "Stn_Press"]] = \
            df_meteo[["Temp", "Dew_Point_Temp", "Stn_Press"]].astype(float)

        df_min_max = df_meteo.resample("1D", on="Date/Time")["Temp", "Wind_Spd"].agg(["min", "max"])
        df_min_max.columns = ['_'.join(col) for col in df_min_max.columns]

        # Climate data is by hours, we resample by day
        df_meteo = df_meteo.resample("1D", on="Date/Time").mean().reset_index()
        df_meteo = df_meteo.join(df_min_max, on="Date/Time")

        return df_meteo

    @staticmethod
    def _make_shifted_counts(df, lags_list):
        """
        Produces lagged sequences of shp_count_prev, assuming shp_count_prev is already lagging behind by 1.
        For instance, if shp_count_prev at time t-0 (row t of a shape_key group) is the count value on the previous
        delta_tp days (excluding the current day), this function adds the counts for t-1, t-2, ... t-lags_list[-1] as
        features at time t.
        :param df: (Pandas DataFrame) data frame containing shape_key column and shp_count_prev
        :param lags_list: (list) list of lags to apply to the series shp_count_prev
        :return: df: (Pandas DataFrame) same data frame but with the lagged sequences as new features.
        """
        temp = pd.DataFrame()
        grouped_counts = df.groupby("shape_key").shp_count_prev
        for ind_lag, lag in enumerate(lags_list):
            temp["shp_count_lag_" + str(ind_lag)] = grouped_counts.shift(lag)
        return df.join(temp)

    # ---
    # Computing the slope on moving windows didn't improve our model so we remove it
    # ---
    # @staticmethod
    # def _diff_func(x):
    #     """
    #     Computes the slope from x's ends
    #     :param x: (Numpy array) array representing current rolling window
    #     :return:
    #     """
    #     x = x[np.where(~np.isnan(x))]
    #     if x.shape[0] >= 1:
    #         return (x[-1] - x[0]) / x.shape[0]
    #     else:
    #         return np.nan
    # ---
    # Computing a rolling ARIMA forecast didn't improve our model and was *very* time consuming so we remove it
    # ---
    # @staticmethod
    # def _arima_pred(x):
    #     """
    #
    #     :param x: (Numpy array) array representing current rolling window
    #     :return:
    #     """
    #     x = x[np.where(~np.isnan(x))]
    #     if x.shape[0] > 1:
    #         est = ARIMA(data=x, ar=1, ma=1, integ=0, family=Normal)
    #         est.fit(method="MLE")
    #         return est.predict(h=1).values
    #     else:
    #         return np.nan

    @staticmethod
    def _group_rolling(inp):
        """
        Wrapper function that applies a function on a rolling window on the feature shp_count_prev for a given
        group  of a grouped data frame
        :param inp: (list) list that contains 3 elements :
                           * (Pandas DataFrame) a group of a grouped data frame (passed by a parallel map)
                           * (int) a time window in days on which to apply the rolling function
                           * (function object or string) callable that can be passed to Pandas aggregate function
        :return: (Pandas Series) the result of applying the function that was passed on a rolling window on the group
        """
        group = inp[0]
        a_period = inp[1]
        a_func = inp[2]
        return group.shp_count_prev.rolling(a_period).aggregate(a_func)

    @staticmethod
    def _prll_make_moving_features(df, time_ranges, funcs):
        """
        Function that creates statistics on moving/sliding windows using parallel computation.
        :param df: (Pandas DataFrame) data frame to which features should be added. Should contain the
                                      shp_count_prev column.
        :param time_ranges: (list) list that contains the length in days of the rolling windows to use
        :param funcs: (list of strings or function objects) list of callables that will be passed to Pandas' aggregate
                                                            function
        :return: (Pandas DataFrame) data frame with the new features
        """
        for ind_per, a_period in enumerate(time_ranges):
            for a_func in funcs:
                if type(a_func) is not str:
                    temp_name = a_func.__name__[1:]
                    col_name = "mv_" + temp_name + "_p_" + str(ind_per)
                else:
                    col_name = "mv_" + a_func + "_p_" + str(ind_per)
                # Compute estimator func on window
                with Pool(cpu_count()) as p:
                    ret_list = p.map(DataSet._group_rolling, [(group, a_period, a_func) for name, group in df. \
                                     set_index(["shape_key", "call_date_time"]).groupby("shape_key")])
                    p.terminate()

                # SHP_COUNT_PREV IS ALREADY LAGGED
                df_temp = pd.concat(ret_list).reset_index().rename(columns={"shp_count_prev": col_name})
                df = df.merge(df_temp, on=["shape_key", "call_date_time"], how="left")

        return df

    @staticmethod
    def _group_expanding(inp):
        """
        Wrapper function that applies a function on an expanding window on the feature shp_count_prev for a given
        group of a grouped data frame
        :param inp: (list) list that contains 2 elements :
                           * (Pandas DataFrame) a group of a grouped data frame (passed by a parallel map)
                           * (function object or string) callable that can be passed to Pandas aggregate function
        :return: (Pandas Series) the result of applying the function that was passed on an expanding window on the group
        """
        group = inp[0]
        a_func = inp[1]
        return group.shp_count_prev.expanding().agg(a_func)

    @staticmethod
    def _make_cumul_features(df, funcs):
        """
        Function that creates statistics on an expanding window using parallel computation.
        :param df: (Pandas DataFrame) data frame to which features should be added. Should contain the
                                      shp_count_prev column.
        :param funcs: (list of strings or function objects) list of callables that will be passed to Pandas' aggregate
                                                            function
        :return: (Pandas DataFrame) data frame with the new features
        """
        for a_func in funcs:
            # Can't pass list to agg() after expanding() so use a for loop
            if type(a_func) is not str:
                temp_name = a_func.__name__[1:]
                col_name = "shp_cumul_" + temp_name
            else:
                col_name = "shp_cumul_" + a_func
            # Compute the cumulative <estimator> counts by day for each shape.
            # SHP_COUNT_PREV IS ALREADY LAGGED
            # Compute estimator func on window
            with Pool(cpu_count()) as p:
                ret_list = p.map(DataSet._group_expanding, [(group, a_func) for name, group in df. \
                                 set_index(["shape_key", "call_date_time"]).groupby("shape_key")])
                p.terminate()

            # SHP_COUNT_PREV IS ALREADY LAGGED
            df_temp = pd.concat(ret_list).reset_index().rename(columns={"shp_count_prev": col_name})
            df = df.merge(df_temp, on=["shape_key", "call_date_time"], how="left")
        return df

    @staticmethod
    def _make_hist_features(df, funcs):
        """
        Function that creates *historical* statistics on an expanding window using parallel computation.
        :param df: (Pandas DataFrame) data frame to which features should be added. Should contain the
                                      shp_count_prev column.
        :param funcs: (list of strings or function objects) list of callables that will be passed to Pandas' aggregate
                                                            function
        :return: (Pandas DataFrame) data frame with the new features
        """
        # This one changes the order of the data frame, so we have to work on a copy
        temp = df[["shape_key", "day_of_year", "year", "shp_count_prev"]]. \
            sort_values(["shape_key", "day_of_year", "year"]).reset_index(drop=True)
        for a_func in funcs:
            if type(a_func) is not str:
                temp_name = a_func.__name__[1:]
                col_name = "shp_hist_" + temp_name
            else:
                col_name = "shp_hist_" + a_func

            # Compute estimator func on window
            with Pool(cpu_count()) as p:
                ret_list = p.map(DataSet._group_expanding, [(group, a_func) for name, group in df. \
                                 set_index(["shape_key", "call_date_time"]).groupby(["shape_key", "day_of_year"])])
                p.terminate()
            # SHP_COUNT_PREV IS ALREADY LAGGED
            df_temp = pd.concat(ret_list).reset_index().rename(columns={"shp_count_prev": col_name})
            df = df.merge(df_temp, on=["shape_key", "call_date_time"], how="left")

        return df

    @staticmethod
    def _group_ewm(inp):
        """
        Wrapper function that applies a function on an exponential weighted moving window on the feature
        shp_count_prev for a given group  of a grouped data frame
        :param inp: (list) list that contains 3 elements :
                           * (Pandas DataFrame) a group of a grouped data frame (passed by a parallel map)
                           * (int) a time window in days on which to apply the rolling function
                           * (function object or string) callable that can be passed to Pandas aggregate function
        :return: (Pandas Series) the result of applying the function that was passed on a rolling window on the group
        """
        group = inp[0]
        an_alpha = inp[1]
        a_func = inp[2]
        return group.shp_count_prev.ewm(alpha=an_alpha).agg(a_func)

    @staticmethod
    def _prll_make_ewm_features(df, alpha_values, funcs):
        """
        Function that creates statistics on an exponential weighted moving window using parallel computation.
        :param df: (Pandas DataFrame) data frame to which features should be added. Should contain the
                                      shp_count_prev column.
        :param alpha_values: (list) list of floats that contains alpha values to use in the EMW. 0 < alpha <= 1.
        :param funcs: (list of strings or function objects) list of callables that will be passed to Pandas' aggregate
                                                            function
        :return: (Pandas DataFrame) data frame with the new features
        """
        for ind_per, an_alpha in enumerate(alpha_values):
            for a_func in funcs:
                if type(a_func) is not str:
                    temp_name = a_func.__name__[1:]
                    col_name = "ewm_" + temp_name + "_p_" + str(ind_per)
                else:
                    col_name = "ewm_" + a_func + "_p_" + str(ind_per)
                # Compute estimator func on window
                with Pool(cpu_count()) as p:
                    ret_list = p.map(DataSet._group_ewm, [(group, an_alpha, a_func) for name, group in df. \
                                     set_index(["shape_key", "call_date_time"]).groupby("shape_key")])
                    p.terminate()
                # SHP_COUNT_PREV IS ALREADY LAGGED
                df_temp = pd.concat(ret_list).reset_index().rename(columns={"shp_count_prev": col_name})
                df = df.merge(df_temp, on=["shape_key", "call_date_time"], how="left")

        return df

    @staticmethod
    def _make_tsfresh_features(df):
        """
        Function that creates new features using the tsfresh package. Check notebook 18 for more details.
        Make sure that tsfresh_settings is a dict that holds the features you want to create. Note that
        shifted counts must be present as features (columns in df) for this function to work (assert below).
        :param df: (Pandas DataFrame) dataframe from which features will be create. 
        :return: (Pandas DataFrame) dataframe with new tsfresh features
        """
        df = df.reset_index()  # we need the index to create "dummy" categories
        lag_cols = [col for col in df.columns if "count_lag" in col]
        # Make sure lag_cols is not empty
        assert (lag_cols)
        lag_cols.extend(["shape_key", "index", "call_date_time"])
        df_temp = df[lag_cols]
        df_temp = df_temp.set_index(["shape_key", "index", "call_date_time"]).stack().reset_index(). \
            drop("level_3", axis=1)
        X = tsfresh.extract_features(df_temp.drop("shape_key", axis=1), column_id="index", column_sort="call_date_time",
                                     column_value=0, chunksize=1, default_fc_parameters=DataSet._tsfresh_settings). \
            reset_index()
        df = df.merge(X, left_on="index", right_on="id").drop(["index", "id"], axis=1)
        return df

    @staticmethod
    def _acti_nom_parser(x):
        # Utility function to parse categories in the 311 data set
        x = x.lower()
        if any(cat in x for cat in DataSet._CAT_0):
            return "0"
        if any(cat in x for cat in DataSet._CAT_1):
            return "1"
        elif any(cat in x for cat in DataSet._CAT_2):
            return "2"
        elif any(cat in x for cat in DataSet._CAT_3):
            return "3"
        else:
            return x

    @staticmethod
    def undiff(X, y):
        """
        Function that computes the inverse of the differentiated counts. It requires that the first count for each group
        has been stored in the DataFrame X (column first_count). It also assumes that X and y are sorted the
        same way such that a simple join on the index can merge them together correctly.
        :param X: (Pandas DataFrame)
        :param y: (Pandas Series) series of counts to undiff
        :return: (Pandas Series) "undiffed" y Series
        """
        assert (np.any(X.columns == "first_count"))
        assert (np.any(X.columns == "shape_key"))
        X = X[["call_date_time", "shape_key", "first_count"]]
        X = X.join(y.to_frame("y"), how="left")
        X = X.merge(X.set_index(["call_date_time", "shape_key"]).groupby("shape_key").y.cumsum().to_frame("y_cumsum"). \
                    reset_index(), how="left", on=["shape_key", "call_date_time"])
        return X.y_cumsum + X.first_count

    # ---
    # Public methods
    # ---

    def __init__(self, shape_path, shape_key, delta_tp=15, delta_tu=30):
        """
        Constructor
        :param shape_path:
        :param shape_key:
        :param delta_tp:
        :param delta_tu:
        """
        # For now, assert that update time is greater than prediction time (we use delta_tu to create the test set)
        assert (delta_tu >= delta_tp)
        self.shape_path = shape_path
        self.shape_key = shape_key
        self.delta_tp = delta_tp
        self.delta_tu = delta_tu

        self.today = pd.to_datetime('today')
        self.db_end_date = None
        self.db_begin_date = None
        self.split_train_test_date = None

        # Check if a data set that matches shape_path, shape_key and delta_t exists
        exists, self.train_parquet_path, self.test_parquet_path = self._dataset_exists()
        if not exists:
            # If no such parquet files exist, call preprocess to create them
            self._preprocess(self.train_parquet_path, self.test_parquet_path)

        self.X_train, self.y_train = DataSet.load_dataset(self.train_parquet_path)
        self.X_test, self.y_test = DataSet.load_dataset(self.test_parquet_path)

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

    def _dataset_exists(self):
        """
        :return:
        """
        # TODO: docstring and improve this func
        try:
            # Check if csv file containing data sets info exists
            csv_file = pd.read_csv(DataSet._save_dataset_path + "dataset_exists.csv")
        except IOError:
            # Otherwise create
            csv_file = pd.DataFrame(columns=["delta_t", "shape_key", "shape_path", "dataset_train_path",
                                             "dataset_test_path"])

        # Check for data sets with our parameters
        dataset = csv_file[(csv_file["delta_t"] == self.delta_tp) & (csv_file["shape_key"] == self.shape_key) &
                           (csv_file["shape_path"] == self.shape_path)]

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
            # Doesn't exists, call preprocess
            # Set up paths to the files
            train_parquet_path = DataSet._save_dataset_path + "dataset_" + str(self.delta_tp) + "_" + str(
                self.shape_key) + "_train.parquet"
            test_parquet_path = DataSet._save_dataset_path + "dataset_" + str(self.delta_tp) + "_" + str(
                self.shape_key) + "_test.parquet"

            # Add info to the csv file so it can be loaded later
            csv_file.append(
                pd.DataFrame([[self.delta_tp, self.shape_key, self.shape_path, train_parquet_path, test_parquet_path]],
                             columns=["delta_t", "shape_key", "shape_path", "dataset_train_path",
                                      "dataset_test_path"])).to_csv(DataSet._save_dataset_path + "dataset_exists.csv",
                                                                    index=False, encoding="utf-8")
            return False, train_parquet_path, test_parquet_path

    def _add_shp_features(self, feature_path, df, df_shp, names):
        """
        Function that loads external data sets and counts features in the shapes defined by df_shp
        :param feature_path: (string) path of the external data set (in parquet format) that will be loaded
        :param df: (Pandas DataFrame) data frame in which features will be added
        :param df_shp: (Pandas DataFrame) data frame containing the shapes which will be used to produce counts
        :param names: (list) list of strings that contain the new features' names
        :return: (Pandas DataFrame) augmented data frame
        """
        # TODO : docstring
        df_feature = pd.read_parquet(feature_path)
        # Create geometry column of Point objects used to join
        df_feature["geometry"] = [Point(xy) for xy in zip(df_feature.Longitude, df_feature.Latitude)]
        df_feature = gpd.GeoDataFrame(df_feature, crs=df_shp.crs)

        # Join the two GeoDataFrames, 'left' and 'within' checks if entries in df (points) are in shape
        df_feature = gpd.tools.sjoin(df_feature, df_shp, how='left', op='within')
        df_feature = df_feature.drop(["index_right", "geometry", "Latitude", "Longitude"], axis=1)

        # Check if there are NaNs, this means some points are not within any shape
        df_feature = df_feature.dropna(axis=0, how="any")
        # Standardize the shape_key code to "shape_key"
        df_feature = df_feature.rename(columns={self.shape_key: "shape_key"})

        df_feature = df_feature.groupby("shape_key").aggregate(["size", "sum"])[names[0]].rename(
            columns={"size": names[1], "sum": names[2]}).reset_index()
        df = df.merge(df_feature, on=["shape_key"], how="left")

        df[names[1:]] = df[names[1:]].fillna(0)

        return df

    def _add_census_features(self, df, df_shp):
        """
        Function that adds the census features to the data frame. We assume that the file at path
        DataSet._census_dataset_path has been generated by notebook 21. The file should contain a "geometry" and a
        "shape_key" column that will be used to merge with df_shp in order to obtain the census features for the shapes
        defined by df_shp.

        This function also assumes that FSA shapes are used as it will load the FSA shapes specified at the path
        DataSet._fsa_shape_path and use DataSet._fsa_shape_key.
        :param df: (Pandas DataFrame) : data frame in which we will append the census features by shape (as constants)
        :param df_shp: (GeoPandas GeoDataFrame) : data frame that contains the shape that will be used to merge with df
        :return: (Pandas DataFrame) : augmented data frame
        """

        # NOTE: we assume that the census data is for FSA shapes, see notebook 21. Since we can't serialize data frames
        # that contain shape (Polygon) objects with parquet, we must load the FSA shapes in this function.
        df_census_geom = pd.read_parquet(DataSet._census_dataset_path)

        # Load FSA shapes to merge with census data
        df_fsa_shp = gpd.GeoDataFrame.from_file(DataSet._fsa_shape_path)
        df_fsa_shp = df_fsa_shp[[DataSet._fsa_shape_key, "geometry"]]

        # We now merge the census data with the FSA shapes. Next we will merge the FSA shapes with the user defined
        # shapes.
        df_census_geom = df_fsa_shp.merge(df_census_geom, left_on=DataSet._fsa_shape_key, right_on="Geo_Code",
                                          how="left").drop("Geo_Code", axis=1)

        # First we compute the overlapping area between FSA shapes and the shapes chosen by the user
        df_shp_area = gpd.overlay(df_shp, df_census_geom[["geometry", DataSet._fsa_shape_key]], how='intersection',
                                  use_sindex=True)

        # We compute the intersection area for each shape
        df_shp_area["inter_area"] = df_shp_area.geometry.area
        # Some shape might have multiple intersections, so we groupby and sum
        df_shp_area = df_shp_area.groupby([self.shape_key, DataSet._fsa_shape_key]).inter_area.sum().reset_index()

        # We now compute the total overlapping area by shape to normalize the intersection areas
        df_shp_area = df_shp_area.join(df_shp_area.groupby(self.shape_key).inter_area.sum().to_frame("total_area"),
                                       how="left", on=self.shape_key)

        # Compute overlapping area proportion by shape
        df_shp_area["prop_area"] = df_shp_area.inter_area / df_shp_area.total_area
        df_shp_area = df_shp_area.drop(["inter_area", "total_area"], axis=1)

        # We now join the census data with our shape, the features will be weighted by the overlapping area later
        df_census_geom = gpd.tools.sjoin(df_census_geom, df_shp, how='right', op='intersects')

        # Merge with proportion by shape, on both df_shp's shape code and df_census_geom's shape code
        df_census_geom = df_census_geom.merge(df_shp_area, how="left", on=[self.shape_key, DataSet._fsa_shape_key])

        # Remove useless columns and rows, drop old "shape_key" and set self.shape_key to "shape_key"
        df_census_geom = df_census_geom.drop(["index_left", DataSet._fsa_shape_key], axis=1).reset_index(drop=True)
        # Check if there are NaNs, this means some points are not within any shape
        df_census_geom = df_census_geom.dropna(axis=0, how="any")
        # Standardize the shape_key code to "shape_key"
        df_census_geom = df_census_geom.rename(columns={self.shape_key: "shape_key"})

        # Weight the census data by the proportion of overlapping area by shape
        # Some columns won't be modified
        chng_cols = df_census_geom.drop(["geometry", "shape_key"], axis=1).columns
        # Apply weights to specific columns
        df_census_geom[chng_cols] = df_census_geom[chng_cols].apply(lambda row: row * row.prop_area, axis=1)
        # Drop useless columns
        df_census_geom = df_census_geom.drop(["prop_area", "geometry"], axis=1)
        # Groupby and sum, i.e. compute weighted sum by groups (shapes)
        df_census_geom = df_census_geom.groupby("shape_key").sum().reset_index()

        # Now all we need to do is merge with df to add constant census features for each shape
        return df.merge(df_census_geom, how="left", on="shape_key")

    def _add_311_features(self, df, df_shp):
        """
        Function that adds the 311 features to the data frame. We assume that the file at path
        DataSet._311_dataset_raw_path is the "raw" 311 data set. The file will be preprocessed in this function, as
        opposed to _add_census_features, where some of the preprocessing is done in a notebook. For further details,
        you can look in notebook 12 where this data set has been explored.
        :param df: (Pandas DataFrame) : data frame in which we will append the census features by shape (as constants)
        :param df_shp: (GeoPandas GeoDataFrame) : data frame that contains the shape that will be used to merge with df
        :return: (Pandas DataFrame) : augmented data frame
        """

        # Later on, we will merge on day_of_year so this feature must be present
        assert (np.any(df.columns == "day_of_year"))

        # Load raw 311 data set
        df_311 = pd.read_csv(DataSet._311_url)

        # Drop rows that contain NaNs, they are only a small fraction of the data
        df_311 = df_311.loc[np.logical_not(df_311[["LOC_LONG", "LOC_LAT"]].isnull().any(axis=1)), :]
        # Drop useless columns
        df_311 = df_311.drop(
            ['ID_UNIQUE', 'NATURE', 'TYPE_LIEU_INTERV', 'RUE', 'RUE_INTERSECTION1', 'RUE_INTERSECTION2',
             'LOC_ERREUR_GDT', 'ARRONDISSEMENT', 'ARRONDISSEMENT_GEO', 'LIN_CODE_POSTAL', 'PROVENANCE_ORIGINALE',
             'PROVENANCE_TELEPHONE', 'PROVENANCE_COURRIEL', 'PROVENANCE_PERSONNE', 'PROVENANCE_COURRIER',
             'PROVENANCE_TELECOPIEUR', 'PROVENANCE_INSTANCE', 'PROVENANCE_MOBILE', 'PROVENANCE_MEDIASOCIAUX',
             'PROVENANCE_SITEINTERNET', 'LOC_X', 'LOC_Y', "UNITE_RESP_PARENT"], axis=1)

        # Make sure DDS_DATE_CREATION is a date time and sort by time
        df_311["DDS_DATE_CREATION"] = pd.to_datetime(df_311.DDS_DATE_CREATION)
        df_311 = df_311.sort_values("DDS_DATE_CREATION").reset_index(drop=True)

        # Parse the fields to create categories
        df_311 = df_311.join(df_311.ACTI_NOM.apply(DataSet._acti_nom_parser).to_frame("ACTI_NOM_parsed")).drop(
            "ACTI_NOM", axis=1)
        # Remove entries that didn't match any categories. Check notebook 12 for more details.
        df_311 = df_311.loc[df_311.ACTI_NOM_parsed.isin(["0", "1", "2", "3"]), :]

        # Create geometry column of Point objects used to join
        df_311["geometry"] = df_311.apply(lambda row: Point(row.LOC_LONG, row.LOC_LAT), axis=1)
        df_311 = gpd.GeoDataFrame(df_311, crs=df_shp.crs)

        # Join the two GeoDataFrames, 'left' and 'within' checks if entries in df_311 (points) are in shape
        df_311 = gpd.tools.sjoin(df_311, df_shp, how='left', op='within')

        # Clean up data frame, remove useless rows/columns
        df_311 = df_311.drop(["index_right", "LOC_LONG", "LOC_LAT"], axis=1)
        # Check if there are NaNs, this means some points are not within any shape
        df_311 = df_311.dropna(axis=0, how="any")
        # Standardize the shape_key code to "shape_key"
        df_311 = df_311.rename(columns={self.shape_key: "shape_key"})

        # We decided to consider the 311 calls by day of the year so it introduces a temporal variation, but it is
        # constant by year.
        df_311["day_of_year"] = df_311.DDS_DATE_CREATION.dt.dayofyear
        # Count number of occurences of each categories in each shape
        df_311 = df_311.groupby(["shape_key", "ACTI_NOM_parsed", "day_of_year"]).size(). \
            to_frame("cat_count").unstack().fillna(0).stack().reset_index()
        # Pivot data frame to have a feature for each count by shape
        df_311 = df_311.pivot_table(index=["shape_key", "day_of_year"], columns="ACTI_NOM_parsed", values="cat_count",
                                    fill_value=0.0)
        # Rename columns by category
        df_311.columns = "count_cat_" + df_311.columns
        # Reset index, get shape_key and day_of_year as features
        df_311 = df_311.reset_index()
        # Now all we have to do is to merge with df
        df = df.merge(df_311, how="left", on=["shape_key", "day_of_year"])
        # In case there are shapes where no event occur, we fill the NaNs with zeros
        df[["count_cat_0", "count_cat_1", "count_cat_2", "count_cat_3"]] = \
            df.loc[:, ["count_cat_0", "count_cat_1", "count_cat_2", "count_cat_3"]].fillna(0.0)
        return df

    def _make_neighbors_dict(self, df_shp):
        """
        This function computes which shape share borders with each other.
        :param df_shp: (GeoPandas data frame) data frame that contains the shapes used
        :return: (dict) dictionary that contains lists of the neighboring shapes' keys for each shape key. Keys are
                        shape keys and values are lists of shape keys.
        """
        neighbor_dict = {}
        # For each shape in df_shp
        for ind, row in df_shp.iterrows():
            # Find which shapes share borders
            sel = df_shp.geometry.touches(row.geometry)
            if sel.any():
                # Fill the dict with shape key values
                neighbor_dict[row.NO_CAS_OP] = df_shp.loc[sel, self.shape_key].values
            else:
                # If the shape has no neighbor, insert empty list
                neighbor_dict[row.NO_CAS_OP] = []
        return neighbor_dict

    def _make_neighbors_features(self, df, df_shp, col_list):
        """
        This function creates new features by computing the mean of some features (defined in col_list) for the shapes
        that share a border (neighbors). For each time step, the neighbors of each shape are selected and the mean of
        the selected features are computed.
        :param df: (Pandas DataFrame) data frame to which features will be added
        :param df_shp: (GeoPandas Data Frame) data frame that contains the shapes used
        :param col_list: (list) list of strings which are feature (column) names.
        :return: (Pandas DataFrame) df augmented with new features
        """

        # Check if features in col_list are actual features in df
        assert set(col_list).issubset(set(df.columns))
        assert np.any(df.columns == "call_date_time")

        # Get the dictionary of neighboring shapes for each shape in df_shp.
        neigh_dict = self._make_neighbors_dict(df_shp)

        def wrap_func(row, group):
            # Wrapper function that is used to compute the mean of the features of the neighbors of row's shape
            neighs = neigh_dict[row.shape_key]
            # Compute the mean of the features in col_list on the shapes that are neighbors to row's shape
            temp = group.loc[group.shape_key.isin(neighs), col_list].mean()
            # Some shapes have no neighbors, call fillna
            temp = temp.fillna(0.0)
            # Add prefix to new features
            temp.index = ["neighs_mean_" + name for name in temp.index.values]
            # Append to current row
            row = row.append(temp)
            return row

        ret_list = []
        # For each time step
        for _, group in df.groupby("call_date_time"):
            # We apply wrap_func on each shape (row)
            ret_list.append(group.apply(lambda x: wrap_func(x, group), axis=1))
        # The new features have been appended to each group, so we just have to concat
        df = pd.concat(ret_list)
        # The order might have been changed, so we reset the index and sort
        df = df.reset_index(drop=True).sort_values(by=["call_date_time", "shape_key"])
        return df

    def _preprocess(self, train_save_path, test_save_path):
        """
        :param train_save_path:
        :param test_save_path:
        :return:
        """
        # TODO: docstring

        # Read local data set, to update, call update_interim_dataset
        try:
            df = pd.read_parquet(DataSet._interim_parquet_path)
        except IOError:
            print("%s not found, running DataSet.update_dataset()\n" % DataSet._interim_parquet_path)
            DataSet.update_dataset()
            df = pd.read_parquet(DataSet._interim_parquet_path)

        df = df.sort_values(by="call_date_time")

        # Useful to uncomment for debugging 
        # df = df.loc[df.call_date_time.dt.year <= 2013, :]

        # Load .shp file in GeoDataFrame
        df_shp = gpd.GeoDataFrame.from_file(self.shape_path)
        df_shp = df_shp[[self.shape_key, "geometry"]]

        # Create geometry column of Point objects used to join
        df["geometry"] = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        df = gpd.GeoDataFrame(df, crs=df_shp.crs)

        # Join the two GeoDataFrames, 'left' and 'within' checks if entries in df (points) are in shape
        df = gpd.tools.sjoin(df, df_shp, how='left', op='within')
        df = df.drop(["index_right", "geometry", "latitude", "longitude"], axis=1)

        # Check if there are NaNs, this means some points are not within any shape
        df = df.dropna(axis=0, how="any")

        # Standardize the shape_key code to "shape_key"
        df = df.rename(columns={self.shape_key: "shape_key"})

        # It seems there are about the same number of calls during the day shift and the night shift. For now, we won't
        # consider the shift in our model. Refer to notebook 07.
        # df["day_shift"] = ((df.call_date_time.dt.hour >= 7) & (df.call_date_time.dt.hour < 17)).astype(int)

        # For each shape, create time entries on each day. Unstacking allows to fill the missing values if mismatch in
        # begin and end dates. This sorts by shape_key + call_date_time and is CRUCIAL.
        # ** Note that running this line twice will result in an error as the data frame will now be filled with entries
        # for every combination shape key + time stamp **
        df = df.groupby("shape_key").resample("d", on="call_date_time").size().to_frame("shp_count") \
            .unstack().fillna(0).stack().reset_index()

        # Compute prediction on the next delta_t days. We store the "undiffed" (normal) value to be able to
        # undo the differentiation later on.
        # Setting time as and grouping by shape key allows us to apply a rolling window of "delta_tp" days
        # on the counts by day for each shape and then to apply the shift function to obtain the counts
        # for the next "delta_tp" days at time t. Merging on shape_key + call_date_time ensures that
        # the new features are correctly added.
        df = df.merge(df.set_index("call_date_time").groupby("shape_key").shp_count.rolling(self.delta_tp).sum(). \
                      shift(-self.delta_tp + 1).to_frame("shp_count_pred_undiff").reset_index(), how="left",
                      on=["shape_key", "call_date_time"])

        # Compute the difference of elements t and t-1 by shape key for each time stamp. This is the target
        # that we wish to predict. We use the same trick as before, but since we are not applying the rolling
        # function, we also have to store shape_key in the index to be able to merge on those features 
        # afterwards. We use the diff() function to differentiate the series.
        df = df.merge(df.set_index(["call_date_time", "shape_key"]).groupby("shape_key").shp_count_pred_undiff. \
                      diff(1).to_frame("shp_count_pred").reset_index(), how="left",
                      on=["shape_key", "call_date_time"])

        # Since "shp_count_pred" represents the counts of the next "delta_tp" days at time t and is already
        # differentiated, we merely have to shift it "delta_tp" days later to access past values at time t
        # without leaking information about the future. We also use the index trick to store call_date_time
        # and shape_key to safely merge into our data frame.
        df = df.merge(df.set_index(["call_date_time", "shape_key"]).groupby("shape_key").shp_count_pred. \
                      shift(self.delta_tp).to_frame("shp_count_prev").reset_index(), how="left",
                      on=["shape_key", "call_date_time"])

        # TODO: leaving it to check if it's ok
        # df = df.drop("shp_count_pred_undiff", axis=1)

        # Add time features
        df = df.join(pd.DataFrame({"year": df.call_date_time.dt.year,
                                   "month": df.call_date_time.dt.month,
                                   "day_of_year": df.call_date_time.dt.dayofyear,
                                   "cos_day_year": np.cos((df.call_date_time.dt.dayofyear - 1) * 2 * np.pi / 365),
                                   "cos_month": np.cos(df.call_date_time.dt.month * 2 * np.pi / 12),
                                   "sin_day_year": np.sin((df.call_date_time.dt.dayofyear - 1) * 2 * np.pi / 365),
                                   "sin_month": np.sin(df.call_date_time.dt.month * 2 * np.pi / 12),
                                   "day_of_week": df.call_date_time.dt.dayofweek,
                                   "weekend": ((df.call_date_time.dt.dayofweek == 0) |
                                               (df.call_date_time.dt.dayofweek == 5) |
                                               (df.call_date_time.dt.dayofweek == 6)).astype(int)
                                   }))

        # Code holidays Cyril
        quebec_holidays = holidays.Canada(prov="QC")
        df["holiday"] = df["call_date_time"].map(lambda x: x in quebec_holidays).astype(int)

        # Create moving and cumulative counts based on simple estimators (mean, median, ...)
        if self.delta_tp > 1:
            time_ranges = [int(self.delta_tp), int(self.delta_tp * 3.0), int(self.delta_tp * 7.0)]
            lags_list = range(0, self.delta_tp)
        else:
            time_ranges = [int(self.delta_tp * 3), int(self.delta_tp * 7), int(self.delta_tp * 15)]
            lags_list = range(0, 15 * self.delta_tp)
        funcs = ["mean", "median", "var", "max", "min"]
        df = DataSet._prll_make_moving_features(df, time_ranges, funcs)
        df = DataSet._make_cumul_features(df, funcs)
        df = DataSet._make_hist_features(df, funcs)

        # ---
        # The EWM mean and var didn't improve our model, so we remove it
        # ---
        # alpha_values = [0.25, 0.5, 0.75]
        # funcs = ["mean", "var"]  # So apparently EWM doesn't work with median, max and min
        # df = DataSet._prll_make_ewm_features(df, alpha_values, funcs)

        # Add previous counts as features for the current time, to be consistent include we include value at t (now)
        df = DataSet._make_shifted_counts(df, lags_list=lags_list)

        # Remove useless column (counts by day)
        df = df.drop("shp_count_prev", axis=1)

        # First and last dates
        self.db_end_date = df.call_date_time.max()
        self.db_begin_date = df.call_date_time.min()

        # To be able to undo the differentiation, we have to keep track of the first count for each shape
        # before applying diff. In order to do that, we create a new column "first_count" which contains
        # the first count of the time serie for each shape.
        # Since we will remove rows containing NaNs in a few steps -- some of which are at the top of the
        # data frame --, we have to look at the latest count for each shape where any column contained a NaN.
        # The only columns that contain NaN at the bottom are "shp_count_pred" and "shp_count_pred_undiff",
        # so we simply drop them before finding the NaNs. This "first_count" value is a constant by shape,
        # so we just need to merge on shape_key.        
        df = df.merge(df.loc[df.drop(["shp_count_pred", "shp_count_pred_undiff"], axis=1).isnull().any(axis=1),
                             ["shape_key", "shp_count_pred_undiff"]].groupby("shape_key").last().reset_index(). \
                      rename(columns={"shp_count_pred_undiff": "first_count"}), how="left", on="shape_key")

        # Entries with time lag have NaN values which correspond to incomplete/false entries, we must remove them
        df = df.dropna(axis=0, how="any")

        # ---
        # Leave this commented, it doesn't seem to improve the model much.
        # ---
        # col_list = ["shp_cumul_mean", "shp_hist_mean", "shp_cumul_var", "shp_hist_var", "mv_mean_p_0",
        #             "mv_mean_p_1", "mv_mean_p_2", "mv_var_p_0", "mv_var_p_1", "mv_var_p_2"]
        # df = self._make_neighbors_features(df, df_shp, col_list)

        # Add factories features
        df = self._add_shp_features("../../data/processed/usines.parquet", df, df_shp,
                                    names=["SUPERFICIE_TERRAIN", "nb_usines", "total_superficie_usines"])
        # Add schools features
        df = self._add_shp_features("../../data/processed/ecoles.parquet", df, df_shp,
                                    names=["SUPERFICIE_TERRAIN", "nb_ecoles", "total_superficie_ecoles"])
        # Add seniors residences features
        df = self._add_shp_features("../../data/processed/residences_personnes_agees.parquet", df, df_shp,
                                    names=["SUPERFICIE_TERRAIN", "nb_resid", "total_superficie_resid"])

        # Add parks features
        df = self._add_shp_features("../../data/processed/parcs.parquet", df, df_shp,
                                    names=["SUPERFICIE_TERRAIN", "nb_parcs", "total_superficie_parcs"])

        df["shape_key"] = pd.to_numeric(df["shape_key"])

        # Add census features
        df = self._add_census_features(df, df_shp)

        df = self._add_311_features(df, df_shp)

        df = DataSet._make_tsfresh_features(df)

        # Get climate data and join dataframes
        df = df.merge(DataSet._get_meteo(self.db_begin_date, self.db_end_date),
                      left_on="call_date_time", right_on="Date/Time", how="left").drop("Date/Time", axis=1)

        # Split in test and train sets, test on delta_tu (>= delta_tp)
        self.split_train_test_date = self.db_end_date - pd.Timedelta(self.delta_tu - 1, unit="d")

        # Sometimes shape_key is typed as an object...
        df["shape_key"] = pd.to_numeric(df["shape_key"])

        # df = df.sort_values(["call_date_time", "shape_key"])

        # Split in test and train sets, test on delta_tu (>= delta_tp)
        test = df.loc[df.call_date_time >= self.split_train_test_date, :]
        train = df.loc[df.call_date_time < self.split_train_test_date, :]

        # As was done previously, we compute the first count to be able to undiff the series.
        # This is necessary since there is a split between the train and test sets.
        test = test.drop("first_count", axis=1)
        test = test.merge(train[["shape_key", "shp_count_pred_undiff"]].groupby("shape_key").last().reset_index(). \
                          rename(columns={"shp_count_pred_undiff": "first_count"}), how="left", on="shape_key")

        # The last "delta_t" days have "incomplete" predictions, remove those rows from train set
        # Note that the last rows of the test set were already removed by the previous dropna()
        end_date = train.call_date_time.max() - pd.Timedelta(self.delta_tp, "d")
        train = train.loc[train.call_date_time <= end_date, :]

        # Save to .parquet
        train.reset_index(drop=True).to_parquet(train_save_path)
        test.reset_index(drop=True).to_parquet(test_save_path)

        return


if __name__ == "__main__":
    # Create interim parquet file that is used by the preprocess method
    # DataSet.create_interim_dataset()
    from timeit import default_timer as timer

    shape_path = "../../data/processed/Sect_cas_op/Sect_cas_op.shp"
    shape_key = "NO_CAS_OP"
    delta_tp = 1
    delta_tu = 30

    start = timer()
    data = DataSet(shape_path, shape_key, delta_tp=delta_tp, delta_tu=delta_tu)
    end = timer()
    print(end - start)

    X_train = data.X_train
    y_train = data.y_train
    z_train = DataSet.undiff(X_train, y_train)
    print((z_train == X_train.shp_count_pred_undiff).all())

    X_test = data.X_test
    y_test = data.y_test
    z_test = DataSet.undiff(X_test, y_test)
    print((z_test == X_test.shp_count_pred_undiff).all())

    splits = data.tscv(5)
    cdt = data.X_train.call_date_time
    print(data.X_train.columns.values)
    for split in splits:
        train_beg = cdt.iloc[split[0][0]]
        train_end = cdt.iloc[split[0][-1]]
        print("train beginning = %s\ntrain end = %s\n" % (train_beg, train_end))
        test_beg = cdt.iloc[split[1][0]]
        test_end = cdt.iloc[split[1][-1]]
        print("test beginning = %s\ntest end = %s\n" % (test_beg, test_end))
