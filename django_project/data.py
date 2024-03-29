from django.conf import settings
import pandas as pd
import numpy as np
import sqlite3 as sql
import requests as requests
import os
import pickle as pkl
import lxml
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import timedelta
from bs4 import BeautifulSoup

location_geocode_data = settings.LOCATION_GEOCODE_DATA
hourly_weather_variables = settings.HOURLY_WEATHER_VARIABLES
 
def balance_non_carbon_generation(data, row_number, generation_type, excess):
    if excess > data.iloc[row_number, data.columns.get_loc(generation_type)]:
        remaining_excess = excess - data.iloc[row_number, data.columns.get_loc(generation_type)]
        data.iloc[row_number, data.columns.get_loc(generation_type)] = 0
    else:
        remaining_excess = 0
        data.iloc[row_number, data.columns.get_loc(generation_type)] = data.iloc[row_number, data.columns.get_loc(generation_type)] - excess
    return data, remaining_excess


def balance_carbon_generation(data, row_number, excess):
    data.iloc[row_number, data.columns.get_loc('carbon')] = np.abs(excess)
    remaining_excess = 0
    return data, remaining_excess


def create_scenerio_data(data, plan, group_by='month'):
    renewable_sources = ['wind(offshore)', 'wind(onshore)', 'solar', 'hydro']
    renewable_sources_plus_battery = ['wind(offshore)', 'wind(onshore)', 'solar', 'hydro', 'battery_supplied']
    carbon_sources = ['coal', 'ccgt']
    carbon = ['carbon']
    other_sources = ['nuclear', 'biomass']
    
    # Calculate the new capacity
    for column, capacity_multiplier in plan['capacity'].items():
        data[column] = data[column] * capacity_multiplier
    
    # Set the battery storage capacity for this scenario
    # The data is in half hour intervals. All power figures are in KWh. Show battery capacity by KW half hours (KWhh)
    storage_capacity_GWhh = plan['storage_capacity_GWh'] * 1000 * 2
    
    data['battery_supplied'] = 0
    data['wasted_GWhh'] = 0
    
    excess_data = np.array(data[renewable_sources + other_sources].sum(axis=1) - data['demand'])
    stored_GWhh = 0
    wasted_GWhh = []
    battery_supplied = []
    excess_generated = []
    
    for excess in np.nditer(excess_data):
        if excess > 0:
            excess_generated.append(excess)
            battery_supplied.append(0)
            stored_GWhh += excess
            if stored_GWhh > storage_capacity_GWhh:
                wasted_GWhh.append(-(stored_GWhh - storage_capacity_GWhh))
                stored_GWhh = storage_capacity_GWhh
            else:
                wasted_GWhh.append(0)
        elif stored_GWhh > np.abs(excess):
            excess_generated.append(0)
            wasted_GWhh.append(0)
            battery_supplied.append(np.abs(excess))
            stored_GWhh = stored_GWhh - np.abs(excess)
        else:
            excess_generated.append(excess + stored_GWhh)
            wasted_GWhh.append(0)
            battery_supplied.append(np.abs(stored_GWhh))
            stored_GWhh = 0
    
    # Add all the hourly trade in excess production and release to the dataset
    data['battery_supplied'] = battery_supplied
    data['wasted_GWhh'] = wasted_GWhh
    data['excess_generated'] = excess_generated
    
    # Default all carbon generation to 0 before calculating how much of this needs to be generated in the new scenario
    data['carbon'] = 0
    
    # Adjust the balance of carbon and non carbon sources based on the new renewable scenario 
    for i, excess in enumerate(data['excess_generated']):
        if excess > 0:
            # balance non carbon sources in order of precedence
            for generation_type in ['nuclear', 'biomass', 'wind(offshore)', 'wind(onshore)']:
                data, excess = balance_non_carbon_generation(data, i, generation_type, excess)
                if excess == 0:
                    break
        else:
            data, excess = balance_carbon_generation(data, i, excess)
    
    return data


# Read and format the windspeed data recorded at different locations around the UK
def format_windspeed_data(location, df):
    df['Date'] = df['time'].str[:10].astype('datetime64[ns]')
    df['Hour'] = df['time'].str[11:13].astype(int)+1
    df.drop('time', axis=1, inplace=True)
    df = df[df['windspeed_10m'].isnull()==False]
    new_columns = [location + '_' + i for i in hourly_weather_variables]
    df.rename(columns=dict(zip(hourly_weather_variables, new_columns)), inplace=True)
    return df


def read_generation_training_data(start_date, generation_type):
    windpowerdata = pd.read_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly.csv'), parse_dates=['date'], usecols=['date','period', generation_type])
    windpowerdata = windpowerdata[windpowerdata['period'].isnull()==False]
    windpowerdata['hour'] = np.ceil(windpowerdata['period'] / 2)
    windpowerdata = windpowerdata.astype({"hour": int})
    windpowerdata = windpowerdata.groupby(['date', 'hour']).agg({generation_type : 'mean'}).reset_index()
    windpowerdata.columns = ['Date', 'Hour', 'Power']
    return windpowerdata


def read_training_data(generation_type):
    # make this a config setting
    # start_date ='2022-08-01'
    # Use only a year (or whole number of years) worth of data, so no one month is over-represented.
    end_date = pd.Timestamp.today()
    start_date = end_date - timedelta(days=365)
    windpowerdata = read_generation_training_data(start_date, generation_type)
    historical_data_url = "https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date="+str(start_date)[:10]+"&end_date="+str(end_date)[:10]+"&hourly=" + ",".join(hourly_weather_variables)
    windspeeddata = {}
    for i, location in enumerate(location_geocode_data[generation_type].items()):
        data = requests.get(historical_data_url.format(lat=str(location[1][0]), lon=str(location[1][1])), verify=False)
        print('Reading data for', location[0], data.status_code)
        windspeeddata[i] = format_windspeed_data(location[0], pd.DataFrame(data.json()['hourly']))
        windpowerdata = windpowerdata.merge(windspeeddata[i], on=['Date', 'Hour'])
    
    windpowerdata.to_csv(os.path.join(settings.DATA_DIR, 'windpowerdata.csv'), index=False)
    return windpowerdata

'''
def read_forecast_data():
    forecast_data_url = "http://api.weatherapi.com/v1/forecast.json?key=c6d909ccb3044b41819172252232907&q={lat},{lon}&days=1"
    forecast_data = pd.DataFrame()
    for i, location in enumerate(location_geocode_data[generation_type].items()):
        jsondata = requests.get(forecast_data_url.format(lat=str(location[1][0]), lon=str(location[1][1])))
        print(location[0], jsondata.status_code)
        forecast_temp = pd.DataFrame()
        for i in jsondata.json()['forecast']['forecastday']:
            forecast_temp = pd.concat([forecast_temp, pd.DataFrame(i['hour'])])
            forecast_temp = forecast_temp.iloc[:,[1,7,8]]
            forecast_temp.columns = ['time', 'windspeed_10m', 'winddirection_10m']
        
        forecast_temp = format_windspeed_data(location[0], forecast_temp)
        if len(forecast_data) == 0:
            forecast_data = forecast_temp[['Date', 'Hour']]
        forecast_data = forecast_data.merge(forecast_temp, on=['Date', 'Hour'])
    
    return forecast_data
'''

def read_forecast_data_old(generation_type):
    forecast_data_url = 'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&past_days=92&forecast_days=10&hourly=' + ','.join(hourly_weather_variables)
    forecast_data = pd.DataFrame()
    for i, location in enumerate(location_geocode_data[generation_type].items()):
        data = requests.get(forecast_data_url.format(lat=str(location[1][0]), long=str(location[1][1])), verify=False)
        print(location[0], data.status_code)
        forecast_temp = format_windspeed_data(location[0], pd.DataFrame(data.json()['hourly']))
        # Note: The windspeed data from the forecast api show lower windspeeds than through the historical recorded data.
        # There is probably a valid reason for this, but it means the data for the forecast needs adjusting.
        # Scale the forecast windspeed to have the same mean as the actual recorded windspeeds used in the model
        # The multiplier is calculated elsewhere in this notebook
        forecast_temp.iloc[:,0] = forecast_temp.iloc[:,0] * 1.206
        if len(forecast_data) == 0:
            forecast_data = forecast_temp[['Date', 'Hour']]
        forecast_data = forecast_data.merge(forecast_temp, on=['Date', 'Hour'])
    
    print(forecast_data)
    return forecast_data


def merge_power_generation_data(grid_data, ntn_data):
    grid_data = grid_data.fillna(0)
    ntn_data = ntn_data.fillna(0)
    # merge the data sets before the database update
    data = ntn_data.merge(grid_data, on=['date', 'period'])
    # make sure all the columns except for date are cast as integers
    integer_columns = list(data)
    integer_columns.remove('date')
    data[integer_columns] = data[integer_columns].astype(np.int64)
    return data


def xml_to_dataframe(xml, xpath):    
    root = lxml.etree.fromstring(xml)

    # This XPath specifically targets <item> tags that are direct children of <responseList> tags.
    items = root.xpath('//responseList/item')
    data = []

    for item in items:
        row_data = {}
        for child in item.getchildren():
            row_data[child.tag] = child.text
        data.append(row_data)
    data=pd.DataFrame(data)
    data['startTimeOfHalfHrPeriod'] = data['startTimeOfHalfHrPeriod'].astype('datetime64[ns]')
    data['settlementPeriod'] = data['settlementPeriod'].astype(np.int64)
    
    return pd.DataFrame(data)


def update_data(base, update):
    # This will combine two data sets with the same column structure
    # Any existing rows will be replaced based on a match with the index
    base.set_index(['date', 'period'], inplace=True)
    update.set_index(['date', 'period'], inplace=True)
    data = update.combine_first(base)
    return data.reset_index()


def format_ntn_data(ntn_data):
    data_mapping_ntn_columns = {'startTimeOfHalfHrPeriod': 'date',
                    'settlementPeriod': 'period',
                    'coal': 'coal',
                    'ccgt': 'ccgt',
                    'ocgt': 'ocgt',
                    'nuclear': 'nuclear',
                    'oil': 'oil',
                    'wind': 'wind(offshore)',
                    'npshyd': 'hydro',
                    'ps': 'pumped',
                    'biomass': 'biomass',
                    'other': 'other'}
    
    # order ntn data in key order
    ntn_data = ntn_data[list(data_mapping_ntn_columns.keys())]
    # give the data the new column names
    ntn_data.columns = list(data_mapping_ntn_columns.values())
    #ntn_data['date'] = ntn_data['date'].apply(lambda x: str(x)[:4] + str(x)[4:6] + str(x)[6:8]).astype('datetime64[ns]')
    return ntn_data


def format_grid_data(grid_data):
    # map the grid data to the database table columns                
    data_mapping_grid_columns = {'SETTLEMENT_DATE': 'date',
                    'SETTLEMENT_PERIOD': 'period',
                    'EMBEDDED_WIND_GENERATION':'wind(onshore)',
                    'EMBEDDED_SOLAR_GENERATION':'solar',
                    'PUMP_STORAGE_PUMPING':'pumped_storage_pumping',
                    'IFA_FLOW': 'ifa',
                    'IFA2_FLOW': 'ifa2',
                    'BRITNED_FLOW': 'britned',
                    'MOYLE_FLOW': 'moyle',
                    'EAST_WEST_FLOW': 'ewic',
                    'NEMO_FLOW': 'nemo'
                    #'NSL_FLOW': 'nsl',
                    #'ELECLINK_FLOW': 'eleclink'
                }
    # order grid data in key order
    grid_data = grid_data[list(data_mapping_grid_columns.keys())]
    # give the data the new column names
    grid_data.columns = list(data_mapping_grid_columns.values())
    #grid_data['date'] = grid_data['date'].astype('datetime64[ns]')
    return grid_data


def read_power_generation_data():

    # Check if the historical data file is already saved to pickle and read
    ntn_data_base = pd.read_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly_ntn.csv'), parse_dates=['date'], index_col=False)
    grid_data_base = pd.read_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly_grid.csv'), parse_dates=['date'], index_col=False)
    
    # Call the external data for the updates to the ntn and grid data
    ntn_data = xml_to_dataframe(requests.get("https://www.bmreports.com/bmrs/?q=ajax/xml_download/FUELHH/xml/").content, "//responseList/item")
    ntn_data = format_ntn_data(ntn_data)
    
    grid_data = pd.read_csv("https://data.nationalgrideso.com/backend/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv", parse_dates=['SETTLEMENT_DATE'])
    grid_data = format_grid_data(grid_data)
    
    # Add the new data to the base data for each set
    ntn_data = update_data(ntn_data_base, ntn_data)
    ntn_data.to_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly_ntn.csv'), index=False)
    grid_data = update_data(grid_data_base, grid_data)
    grid_data.to_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly_grid.csv'), index=False)
    
    # Save the final data set with all the merged data
    generation_half_hourly = merge_power_generation_data(grid_data, ntn_data)    
    generation_half_hourly.to_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly.csv'), index=False)

    #generation_half_hourly = pd.read_csv(os.path.join(settings.DATA_DIR, 'generation_half_hourly.csv'), parse_dates=['date'], index_col=False)
    
    return generation_half_hourly
