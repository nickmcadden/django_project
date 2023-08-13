import pandas as pd
import numpy as np
import requests as requests
import os
from django.conf import settings


# Read the historical wind speed data from some locations around the UK
location_geocode_data = {'Bracknell' : (51.4136, -0.7505), 
                         'Cardiff': (51.48, -3.18),
                         'Leeds' : (53.7965, -1.5478),
                         'Belfast': (54.5968, -5.9254),
                         'Edinburgh': (55.9521, -3.1965),
                         'Inverness': (57.4791, -4.224),
                         'Norwich': (52.6278, 1.2983),
                         'Hull': (53.7446, -0.3352),
                         'Carlisle': (54.8951, -2.9382)}

# Read and format the windspeed data recorded at different locations around the UK
def format_windspeed_data(location, df):
    df['Date'] = df['time'].str[:10].astype('datetime64[ns]')
    df['Hour'] = df['time'].str[11:13].astype(int)+1
    df.drop('time', axis=1, inplace=True)
    df = df[df['windspeed_10m'].isnull()==False]
    df.rename(columns={"windspeed_10m": location+"_"+"windspeed_10m", 
                       "winddirection_10m": location+"_"+"winddirection_10m",}, inplace=True)
    return df

def read_wind_power_data(start_date):
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'GenerationbyFuelType_20220701_to_present.csv')
    windpowerdata = pd.read_csv(csv_path, parse_dates=['Date'], usecols=['Date', 'HalfHourPeriod', 'Wind'])
    windpowerdata = windpowerdata[windpowerdata['HalfHourPeriod'].isnull()==False]
    windpowerdata['Hour'] = np.ceil(windpowerdata['HalfHourPeriod'] / 2)
    windpowerdata = windpowerdata.astype({"Hour": int})
    windpowerdata = windpowerdata.groupby(['Date', 'Hour']).agg({'Wind' : 'mean'}).reset_index()
    return windpowerdata

def read_training_data(start_date):
    windpowerdata = read_wind_power_data(start_date)
    historical_data_url = "https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date="+str(start_date)+"&end_date=2023-08-01&hourly=windspeed_10m,winddirection_10m"
    
    windspeeddata = {}
    
    for i, location in enumerate(location_geocode_data.items()):
        data = requests.get(historical_data_url.format(lat=str(location[1][0]), lon=str(location[1][1])))
        print('Reading data for', location[0], data.status_code)
        windspeeddata[i] = format_windspeed_data(location[0], pd.DataFrame(data.json()['hourly']))
        windpowerdata = windpowerdata.merge(windspeeddata[i], on=['Date', 'Hour'])
    
    return windpowerdata

def read_forecast_data():
    forecast_data_url = "http://api.weatherapi.com/v1/forecast.json?key=c6d909ccb3044b41819172252232907&q={lat},{lon}&days=1"
    forecast_data = pd.DataFrame()
    
    for i, location in enumerate(location_geocode_data.items()):
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

def read_forecast_data_old(start_date):
    forecast_data_url = 'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&hourly=windspeed_10m,winddirection_10m&past_days=92&forecast_days=1'
    forecast_data = pd.DataFrame()

    for i, location in enumerate(location_geocode_data.items()):
        data = requests.get(forecast_data_url.format(lat=str(location[1][0]), long=str(location[1][1])))
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

    return forecast_data
