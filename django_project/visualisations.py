import re
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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


def uk_wind_power_map():
    # Wind farm map data 
    wind_farms = read_wind_farm_data()#.sort_values('Completion')
    # Latitude and Longitude of a central point in the UK (around Birmingham)
    uk_coords = [59.8, -2.8]

    # Create a map centered around the UK
    uk_map = folium.Map(location=uk_coords, 
                    zoom_start=5,
                    tiles=None,  
                    max_bounds=True
                    )

    # Add markets at each wind farm location
    for i, row in wind_farms.iterrows():
        print(row.latitude, row.Power)
        if row.latitude > 0 and row.Power > 0:
            print([row.latitude, row.longitude], row.Name)
            #folium.Marker( tooltip=row.Name, popup=row.Name).add_to(uk_map)
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=row.Power / 50,
                color="red",
                fill=True,
                fill_color="red",
                tooltip=f"Name: {row.Name}<br>Turbines: {row.Turbines}<br>Completion: {row.Completion}<br>Cost: {row.Cost}"
            ).add_to(uk_map)

    # Overlay your image
    img_bounds = [(49.7, -9.25), (59.00, 3.9)]  # Example bounds, adjust to fit your image

    folium.raster_layers.ImageOverlay(
        image="uk_map_grey.png",
        bounds=img_bounds,
        interactive=True,                
        cross_origin=False,
        zindex=1
    ).add_to(uk_map)

    # Restrict the viewable area to the image bounds and limit zoom
    uk_map.fit_bounds(img_bounds)
    uk_map.options['minZoom'] = 5  # Example, adjust as needed
    uk_map.options['maxZoom'] = 6  # Prevent zooming

    # Save the map to an HTML file
    uk_map.save('wind_farm_map.html')

    folium_header = uk_map.get_root().header.render()
    map_html = uk_map.get_root().html.render()
    map_script = uk_map.get_root().script.render()

    return folium_header, map_html, map_script

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
                        
    print(data)

    print(source_to_category_map, data.columns.values)
    
    # Make a new data frame containing all the chart data
    chartdata = pd.DataFrame({'source': data.columns.values, 
                              'category': [source_to_category_map[i] for i in data.columns.values], 
                              'heading': [heading] * len(source_to_category_map),
                              'power': data.values.flatten(),
                              'source_colour': [source_to_colour_map[i] for i in data.columns.values],
                              'category_colour': [category_to_colour_map[i] for i in [source_to_category_map[j] for j in data.columns.values]]
                              })
    
    print(chartdata)
    
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


def show_model_evaluation(daily_training, daily_forecast):
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_training["Date"], y=daily_training["Wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=daily_forecast["Date"], y=daily_forecast["Forecast_Ensemble"], name='Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="Model Training Performance", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    return fig
    
    
def show_todays_forecast(forecast_today):
    # Create a plot of hourly forecast for today vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_0"], name = 'Forecast 0', line=dict(color='lightgrey', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_1"], name = 'Forecast 1', line=dict(color='lightslategrey', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_2"], name = 'Forecast 2', line=dict(color='lightsteelblue', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Ensemble"], name = 'Ensemble Forecast', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="UK Wind Power Forecast (MW) " + pd.Timestamp.today().strftime("%A %d %B"), showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_xaxes(minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000,), ticklabelmode="period", tickformat="%H:%M%p")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def show_forecast_vs_actual(grid_generation_today, forecast_today):
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_generation_today["hour"], y=grid_generation_today["wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Ensemble"], name = 'Ensemble Forecast', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="UK Wind Power Forecast (MW) " + pd.Timestamp.today().strftime("%A %d %B"), showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_xaxes(minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000,), ticklabelmode="period", tickformat="%H:%M%p")
    fig.update_yaxes(tickformat=",.0f")
    return fig
    
'''
# This is the code to create a cumulative bar chart showing the uk's growth in offshore wind capacity by time
data = data.groupby('Completion').sum().reset_index()
# calculate the cumulative power capacity by month
data['Cumulative_Power'] = np.cumsum(data['Power'])
fig = px.bar(data, x="Completion", y="Cumulative_Power")
# fig.add_trace(go.bar(x=wind_farms["Completion"], y=np.cumsum(wind_farms["Power"]), name = 'Wind Power', bar=dict(color='green', width=2)))
fig.update_layout(title="Wind Farm Project Completions (Cumulative Power Capacity by Month)", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
fig.show()
'''