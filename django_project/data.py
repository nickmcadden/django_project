from django.conf import settings
import pandas as pd
import numpy as np
import requests as requests
import os
import pickle as pkl
import lxml
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup


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
   
    
def merge_power_generation_data(grid_data, ntn_data):
    # map the ntn data to the database table columns
    data_mapping_ntn_columns = {'startTimeOfHalfHrPeriod': 'date',
                    'settlementPeriod': 'period',
                    'coal': 'coal',
                    'ccgt': 'gas',
                    'ocgt': 'ocgt',
                    'nuclear': 'nuclear',
                    'oil': 'oil',
                    'wind': 'wind',
                    'npshyd': 'hydro',
                    'ps': 'pumped',
                    'biomass': 'biomass',
                    'other': 'other'}

    # map the grid data to the database table columns                
    data_mapping_grid_columns = {'SETTLEMENT_DATE': 'date',
                    'SETTLEMENT_PERIOD': 'period',
                    'EMBEDDED_WIND_GENERATION':'embedded_wind',
                    'EMBEDDED_SOLAR_GENERATION':'solar',
                    'PUMP_STORAGE_PUMPING':'pumped_storage_pumping',
                    'IFA_FLOW': 'ifa',
                    'IFA2_FLOW': 'ifa2',
                    'BRITNED_FLOW': 'britned',
                    'MOYLE_FLOW': 'moyle',
                    'EAST_WEST_FLOW': 'ewic',
                    'NEMO_FLOW': 'nemo',
                    'NSL_FLOW': 'nsl',
                    'ELECLINK_FLOW': 'eleclink'}
    
    # order ntn data in key order
    ntn_data = ntn_data[list(data_mapping_ntn_columns.keys())]
    # give the data the new column names
    ntn_data.columns = list(data_mapping_ntn_columns.values())
    # order grid data in key order
    grid_data = grid_data[list(data_mapping_grid_columns.keys())]
    # give the data the new column names
    grid_data.columns = list(data_mapping_grid_columns.values())
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
    

def update_power_generation_data(base, update):
    base.set_index(['date', 'period'], inplace=True)
    update.set_index(['date', 'period'], inplace=True)
    data = base.combine_first(update)
    return data.reset_index()


def read_power_generation_data():
    # Check if the historical data file is already saved to pickle and read
    if os.path.isfile("generation_half_hourly.pickle"):
        print("Base data found")
        generation_half_hourly_base = pkl.load(open("generation_half_hourly.pickle", "rb"))
    else:
        ntn_data = pd.read_csv("GenerationbyFuelType_20220701_to_present.csv", parse_dates=['startTimeOfHalfHrPeriod'])
        grid_data = pd.read_csv("https://data.nationalgrideso.com/backend/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv", parse_dates=['SETTLEMENT_DATE'])
        generation_half_hourly_base = merge_power_generation_data(grid_data, ntn_data)
        generation_half_hourly_base.to_csv('generation_half_hourly_base.csv')

    # Call the external data for the updates to the ntn and grid data
    ntn_data = xml_to_dataframe(requests.get("https://www.bmreports.com/bmrs/?q=ajax/xml_download/FUELHH/xml/").content, "//responseList/item")
    grid_data = pd.read_csv("https://data.nationalgrideso.com/backend/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv", parse_dates=['SETTLEMENT_DATE'])
    generation_half_hourly_update = merge_power_generation_data(grid_data, ntn_data)
    generation_half_hourly_update.to_csv('generation_half_hourly_update.csv')
    
    # Combine the base data set with the new data
    data = update_power_generation_data(generation_half_hourly_base, generation_half_hourly_update)
    data.to_csv('generation_half_hourly.csv')
    
    # Save the new data as a new base data file
    pkl.dump(data, open("generation_half_hourly.pickle", "wb"))
    
    return data


def plot_sunburst_chart(data, heading):
    # Map sources to main categories
    source_to_category_map = {
                        'coal':'carbon',
                        'gas':'carbon',
                        'ocgt':'carbon',
                        'nuclear':'other',
                        'oil':'carbon',
                        'wind':'renewable',
                        'hydro':'renewable',
                        'pumped':'other',
                        'biomass':'other',
                        'other':'other',
                        'solar':'renewable',
                        'pumped_storage_pumping':'other',
                        'ifa':'transfer',
                        'ifa2':'transfer',
                        'britned':'transfer',
                        'moyle':'transfer',
                        'ewic':'transfer',
                        'nemo':'transfer',
                        'nsl':'transfer',
                        'eleclink':'transfer'}

    # Map sources to colours
    source_to_colour_map = {
                       'coal': '#CC0000',
                       'gas': '#CC0033',
                       'ocgt': '#CC0066',
                       'oil': '#CC0099',
                       'nuclear': '#000066',
                       'pumped': '#000099',
                       'biomass': '#0000CC',
                       'other': '#0000FF',
                       'pumped_storage_pumping': '#0000FF',
                       'wind': '#00CC00',
                       'hydro': '#00CC33',
                       'solar': '#FFFF33',
                       'ifa': '#888888',
                       'ifa2': '#888888',
                       'britned': '#888888',
                       'moyle': '#888888',
                       'ewic': '#888888',
                       'nemo': '#888888',
                       'nsl': '#888888',
                       'eleclink':'#888888'}
    
    # Map categories to colours
    category_to_colour_map = {
                        '(?)':'#ffffff',
                         heading: '#ffffff',
                        'carbon': '#AA0000',
                        'other': '#2244AA', 
                        'renewable': '#00CC00',
                        'transfer': '#888888'}
    
    print(len(source_to_category_map))
    print(data.columns.values)
    print(len(data.values.flatten()))
    
    # Make a new data frame containing all the chart data
    chartdata = pd.DataFrame({'source': data.columns.values, 
                              'category': [source_to_category_map[i] for i in data.columns.values], 
                              'heading': [heading] * len(source_to_category_map),
                              'power': data.values.flatten(),
                              'source_colour': [source_to_colour_map[i] for i in data.columns.values],
                              'category_colour': [category_to_colour_map[i] for i in [source_to_category_map[j] for j in data.columns.values]]
                              })
    
    # Calculate percentages of total power generation at source and category level
    chartdata['percentage'] = chartdata['power'] / data.sum(axis=1).sum() * 100
    categories = chartdata.groupby('category').sum().reset_index()
    categories['percentage'] = categories['power'] / data.sum(axis=1).sum() * 100
    label_to_percentage_map = {**dict(zip(chartdata.source,chartdata.percentage)), **dict(zip(categories.category,categories.percentage))}
    label_to_percentage_map[heading] = 100
    
    # The colour maps for source and category need to be combined into one
    combined_colour_map = {**source_to_colour_map, **category_to_colour_map} 

    # plot the chart
    fig_express = px.sunburst(chartdata, 
                    path=['heading', 'category', 'source'], 
                    values='power', color='category', 
                    color_discrete_map=category_to_colour_map,
                    width=400, height=400)

    # Convert to graph_objects figure
    fig_go = go.Figure(fig_express)

    # Update hover labels
    fig_go.update_traces(marker_colors=[combined_colour_map[i] for i in fig_express.data[-1].labels], 
                        customdata=[label_to_percentage_map[i] for i in fig_express.data[-1].labels], 
                        hoverlabel=dict(bgcolor="white"),
                        hoverinfo='none',
                        hovertemplate='<b>Value:</b> %{value}<br><b>Percentage:</b> %{customdata:.2f}%')
    return fig_go


def show_power_generation_chart(data, period):
    # filter the data based on the period
    match period:
        case 'latest':
            data = data[data['date']>=pd.Timestamp.today().floor('D')]
            data = data.loc[data['period'] == data['period'].max()].set_index(['date', 'period'])
        case 'today':
            data = data[data['date']>=pd.Timestamp.today().floor('D')]
        case 'month':
            print("---")
        case 'year':
            print("---")
    
    # Merge the two wind sources
    data['wind'] = data['wind'] + data['embedded_wind']
    data.drop(columns=['embedded_wind'], inplace=True)
    
    # Find the total power and create a heading for the chart
    total_power = data.sum(axis=1).sum()
    chart_heading =  'Latest<br>' + str("{:.1f}".format(total_power / 1000)) + ' GW'
    fig_go = plot_sunburst_chart(data, chart_heading)
    return fig_go
