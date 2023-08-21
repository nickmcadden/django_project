import os
import re
import pandas as pd
import numpy as np
import pickle as pkl
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup


def merge_power_generation_data(grid_data, ntn_data):
    # map the ntn data to the database table columns
    data_mapping_ntn_columns = {'startTimeOfHalfHrPeriod': 'date',
                    'settlementPeriod': 'period',
                    'coal': 'coal',
                    'ccgt': 'ccgt',
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
                    'EMBEDDED_SOLAR_GENERATION':'embedded_solar',
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
    

def xml_to_dataframe(xml, id_tag):
    s = BeautifulSoup(xml, 'xml')
    items = s.find_all('responseList')[0].find_all('item')
    data = []

    for item in items:
        row_data = {}
        for tag in item.find_all():
            row_data[tag.name] = tag.text
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
    ntn_data = xml_to_dataframe(requests.get("https://www.bmreports.com/bmrs/?q=ajax/xml_download/FUELHH/xml/").content, "item")
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
                        'ccgt':'carbon',
                        'ocgt':'carbon',
                        'gas': 'carbon',
                        'nuclear':'other',
                        'oil':'carbon',
                        'wind':'renewable',
                        'solar':'renewable',
                        'hydro':'renewable',
                        'pumped':'other',
                        'biomass':'other',
                        'other':'other',
                        'embedded_solar':'renewable',
                        'pumped_storage_pumping':'other',
                        'ifa':'transfer',
                        'ifa2':'transfer',
                        'britned':'transfer',
                        'moyle':'transfer',
                        'ewic':'transfer',
                        'nemo':'transfer',
                        'nsl':'transfer',
                        'eleclink':'transfer'}

    # Create colors and create colour mappings for category and source
    a, b, c, d = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Greys]

    # Map sources to colours
    source_to_colour_map = {
                       'coal': '#CC0000',
                       'ccgt': '#CC0033',
                       'ocgt': '#CC0066',
                       'gas': '#CC0066',
                       'oil': '#CC0099',
                       'nuclear': '#000066',
                       'pumped': '#000099',
                       'biomass': '#0000CC',
                       'other': '#0000FF',
                       'pumped_storage_pumping': '#0000FF',
                       'wind': '#00CC00',
                       'hydro': '#00CC33',
                       'solar': '#FFFF33',
                       'embedded_solar': '#FFFF33',
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
    # print(fig_go.data)
    fig_go.show()
    return


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
    plot_sunburst_chart(data, chart_heading)
    return
    
    import re


def dms_string_to_decimal(dms_str):
    """
    Convert a string in DMS (Degrees-Minutes-Seconds) format to decimal degrees.
    example:
        
    lat = dms_string_to_decimal("51°28'38\"N")
    lon = dms_string_to_decimal("0°0'12\"W")

    Parameters:
    - dms_str: The string containing the DMS formatted coordinate.

    Returns:
    - Decimal representation of the DMS coordinate.
    """

    # Extracting values using regex
    match = re.match(r'(\d+)°(\d+)' + r"'([\d\.]+)\"([NSEW])", str(dms_str).upper())
    if not match:
        return np.nan

    degrees, minutes, seconds, direction = match.groups()
    degrees, minutes, seconds = float(degrees), float(minutes), float(seconds)

    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees


def read_wind_farm_data():
    wind_farms = pd.read_csv('uk_wind_farms.csv')
    print(wind_farms)
    # Splitting the coordinate column into latitude and longitude
    wind_farms[['latitude_dms', 'longitude_dms']] = wind_farms['Location'].str.split(' ', expand=True)

    print(wind_farms.dtypes)
    # Converting DMS string columns to decimal
    wind_farms['latitude'] = wind_farms['latitude'].apply(dms_string_to_decimal)
    wind_farms['longitude'] = wind_farms['longitude'].apply(dms_string_to_decimal)

    # Drop the intermediate columns if you want
    wind_farms = wind_farms.drop(columns=['latitude_dms', 'longitude_dms'])
    print(wind_farms)
    
read_wind_farm_data()

    
    
