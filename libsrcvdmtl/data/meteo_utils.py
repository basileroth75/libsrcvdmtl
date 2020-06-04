import io
import requests
import pandas as pd


def request_meteo(year, month, stationID):
    """
    Function that calls Meteo Canada's API to extract a weather station's hourly weather data at a given year and
    month.
    :param year: (int) year
    :param month: (int) month
    :param stationID: (int) id of the weather station that you wish to query
    :return: (Pandas DataFrame) data frame that contains the hourly weather data at a given year and month
    """
    # TODO: docstring
    query = '?format=csv&stationID={}&Year={}&Month={}&timeframe=1'.format(stationID, year, month)
    response = requests.get('http://climat.meteo.gc.ca/climate_data/bulk_data_e.html{}'.format(query))
    textio = io.StringIO(response.content.decode('utf-8'))
    ''' Modified by Basile R
    Removed the header=14 parameter because it was not loading the column names from the csv file collected from the API.
    colnames=["Year", "Month", "Day", "Time", "Data Quality", "Temp Flag", "Weather","Dew Point Temp Flag", "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag","Stn Press Flag", "Hmdx Flag", "Wind Chill Flag", "Visibility Flag"]
    table = pd.read_csv(textio, names=colnames, header=None, decimal=',')
    table = pd.read_csv(textio, header=14, decimal=',')
    '''
    table = pd.read_csv(textio, decimal=',')
    return table


def get_meteo(year_begin, year_end):
    """
    Function that returns the weather for all dates between two years. Note that weather data is produced hourly.
    :param year_begin: (int) first year
    :param year_end: (int) last year
    :return: (Pandas DataFrame) data frame that contains the hourly weather data between year_begin and year_end
    """
    st_hubert = 48374
    # Call climate data API
    df_meteo = pd.concat([request_meteo(y, m, stationID=st_hubert)
                          for y in range(year_begin, year_end + 1)
                          for m in range(1, 13)])
    return df_meteo


def get_meteo_at_date(time_stamp):
    """
    Function that only returns the weather for a given date. Note that it doesn't seem possible to only request one day,
    thus we specify a year and a month and afterwards filter out the other dates.
    :param time_stamp: (DateTime object) date on which weather is needed
    :return: (Pandas DataFrame) weather data on a given date
    """
    st_hubert = 48374
    # Call climate data API
    df_meteo = request_meteo(time_stamp.year, time_stamp.month, stationID=st_hubert)
    print(df_meteo)
    return df_meteo.loc[pd.to_datetime(df_meteo["Date/Time"]) == time_stamp]
