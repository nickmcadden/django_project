import os
import re
import pandas as pd
import numpy as np
import pickle as pkl
import requests
import lxml
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup


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
    wind_farms = pd.read_csv('uk_wind_farms.csv', parse_dates=['Completion'])
    # Splitting the coordinate column into latitude and longitude
    wind_farms[['latitude_dms', 'longitude_dms']] = wind_farms['Location'].str.split(' ', expand=True)

    # Converting DMS string columns to decimal
    wind_farms['latitude'] = wind_farms['latitude_dms'].apply(dms_string_to_decimal)
    wind_farms['longitude'] = wind_farms['longitude_dms'].apply(dms_string_to_decimal)

    # Drop the intermediate columns if you want
    wind_farms = wind_farms.drop(columns=['latitude_dms', 'longitude_dms'])
    
    return wind_farms
    
    
def format_windfarm_data(uk_windfarms):
    start = uk_windfarms['Completion'].min()
    end = uk_windfarms['Completion'].max()
    all_months = pd.DataFrame({'Name':'', 'Completion': pd.date_range(start=start, end=end, freq="M"), 'Power':0, 'Colour': 'lightslategray'})
    uk_windfarms['Colour'] = 'crimson'
    uk_windfarms = all_months.merge(uk_windfarms, on=['Name', 'Completion', 'Power', 'Colour'], how='outer')
    uk_windfarms = uk_windfarms.sort_values(by=['Completion', 'Colour'], ascending=True)
    uk_windfarms = uk_windfarms.groupby(['Completion', 'Colour', 'Name']).sum().reset_index()
    uk_windfarms['Cumulative Power'] = np.cumsum(uk_windfarms['Power'])
    uk_windfarms = uk_windfarms[uk_windfarms['Completion']>=pd.Timestamp.today().floor('D')]
    uk_windfarms['Completion'] = uk_windfarms['Completion'].astype(str).apply(lambda x: x[:7])
    uk_windfarms.drop_duplicates(subset=['Completion'], keep='first', inplace=True)
    print(uk_windfarms.dtypes)
    print(uk_windfarms)
    return uk_windfarms


def show_uk_wind_farms():
    uk_windfarms = read_wind_farm_data()
    uk_windfarms = format_windfarm_data(uk_windfarms)
    # calculate the cumulative power capacity by month
    fig = go.Figure(data=[go.Bar(
        x=uk_windfarms['Completion'],
        y=uk_windfarms['Cumulative Power'],
        marker_color=uk_windfarms['Colour'],
        text=uk_windfarms['Name'],
        textposition="outside",
        textangle=45 # marker color can be a single color value or an iterable
    )])
    fig.update_layout(title_text="Offshore Wind Farm Completions (Capacity by Month)")
    fig.update_layout(width=600, height=400)
    return fig


def merge_power_generation_data(grid_data, ntn_data):
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
    data = base.combine_first(update)
    return data.reset_index()


def format_ntn_data(ntn_data):
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
                    'EMBEDDED_WIND_GENERATION':'embedded_wind',
                    'EMBEDDED_SOLAR_GENERATION':'embedded_solar',
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
    ntn_data_base = pd.read_csv("generation_half_hourly_ntn.csv", parse_dates=['date'], index_col=False)
    #ntn_data_base = format_ntn_data(ntn_data_base)
    grid_data_base = pd.read_csv("generation_half_hourly_grid.csv", parse_dates=['date'], index_col=False)
    #grid_data_base = format_grid_data(grid_data_base)
    
    # Call the external data for the updates to the ntn and grid data
    ntn_data = xml_to_dataframe(requests.get("https://www.bmreports.com/bmrs/?q=ajax/xml_download/FUELHH/xml/").content, "//responseList/item")
    ntn_data = format_ntn_data(ntn_data)
    grid_data = pd.read_csv("https://data.nationalgrideso.com/backend/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv", parse_dates=['SETTLEMENT_DATE'])
    grid_data = format_grid_data(grid_data)
    
    # Add the new data to the base data for each set
    ntn_data = update_data(ntn_data_base, ntn_data)
    ntn_data.to_csv('generation_half_hourly_ntn.csv', index=False)
    grid_data = update_data(grid_data_base, grid_data)
    grid_data.to_csv('generation_half_hourly_grid.csv', index=False)
    
    # Save the final data set with all the merged data
    generation_half_hourly = merge_power_generation_data(grid_data, ntn_data)
    generation_half_hourly.to_csv('generation_half_hourly.csv', index=False)
    
    return generation_half_hourly


def show_model_error(chart_data):
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data["forecast_lag"], y=chart_data["absolute_error"], name = 'Model Error', line=dict(color='green', width=2)))
    fig.update_layout(title="Model Error (MW) vs Prediction Time Lag (Hours)", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    return fig


def evaluate_forecast(generation_data):
    if os.path.isfile("saved_forecasts.pickle"):
        print("Saved forecasts found")
        saved_forecasts = pkl.load(open("saved_forecasts.pickle", "rb"))
        saved_forecasts.to_csv('saved_forecasts.csv', index=False)
    else:
        print("Missing forecast data")
    generation_data['forecast_date_hour'] = generation_data['date'] + pd.to_timedelta((generation_data['period']-1)/2, unit='h')    
    generation_data['hour'] = (generation_data['period']-1)/2
    evaluation_data = saved_forecasts.merge(generation_data, left_on=['Date', 'Hour'], right_on=['date', 'hour'])  
    evaluation_data = evaluation_data[evaluation_data['created_at']<=evaluation_data['forecast_date_hour']]
    evaluation_data['forecast_lag'] = np.round((evaluation_data['forecast_date_hour'] - evaluation_data['created_at']).dt.total_seconds() / 3600)
    evaluation_data['absolute_error'] = np.abs(evaluation_data['wind(offshore)'] - evaluation_data['Forecast_Stack']) 
    evaluation_data = evaluation_data[['forecast_date_hour', 'created_at', 'forecast_lag', 'wind(offshore)', 'Forecast_Stack', 'absolute_error']]
    evaluation_data.to_csv('evaluation_data.csv', index=False)
    chart_data = evaluation_data.copy().groupby(['forecast_lag']).agg({'absolute_error':'mean'}).reset_index()
    fig_model_evaluation = show_model_error(chart_data)
    fig_model_evaluation.show()
    return

data = pd.read_csv("generation_half_hourly.csv", parse_dates=['date'], index_col=False)
evaluate_forecast(data)


