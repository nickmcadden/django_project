from django.conf import settings
import re
import os
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
    wind_farms = pd.read_csv(os.path.join(settings.DATA_DIR, 'uk_wind_farms.csv'), parse_dates=['Completion'])
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
        if row.latitude > 0 and row.Power > 0:
            #print([row.latitude, row.longitude], row.Name)
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
        image=os.path.join(settings.DATA_DIR, 'uk_map_grey.png'),
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
    uk_map.save(os.path.join(settings.DATA_DIR, 'wind_farm_map.html'))

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


def format_generation_type_text(generation_type):
    # Creates a text formatted version of the generation type
    match generation_type:
        case 'wind(offshore)':
            text = 'Offshore Wind Power'
        case 'wind(onshore)':
            text = 'Onshore Wind Power'
        case 'solar':
            text = 'Solar Power'
        case 'hydro':
            text = 'Hydro Power'
    return text


def show_power_chart(data, period):
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
    data['wind'] = data['wind(offshore)'] + data['wind(onshore)']
    data.drop(columns=['wind(offshore)', 'wind(onshore)'], inplace=True)
    
    # Find the total power and create a heading for the chart
    total_power = data.sum(axis=1).sum()
    chart_heading =  'Latest<br>' + str("{:.1f}".format(total_power / 1000)) + ' GW'
    fig_go = plot_sunburst_chart(data, chart_heading)
    return fig_go


def show_forecast_vs_actual(grid_generation, generation_type, forecast_today):
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_generation["hour"], y=grid_generation[generation_type], name = format_generation_type_text(generation_type), line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Stack"], name = 'Ensemble Forecast', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_0"], name = 'Neural Network Model', line=dict(color='lightgrey', width=1)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_1"], name = 'Random Forest Model', line=dict(color='lightsteelblue', width=1)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_2"], name = 'XGBoost Model', line=dict(color='lightslategrey', width=1)))
    fig.update_layout(title=format_generation_type_text(generation_type) + " Forecast (MW) " + pd.Timestamp.today().strftime("%A %d %B"), showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.99))
    fig.update_xaxes(nticks=10)
    fig.update_layout(hovermode="x unified")
    return fig

    
def show_forecast_models(forecast_today):
    # Create a plot of hourly forecast for today vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Stack"], name = 'Ensemble Forecast', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_0"], name = 'Neural Network Model', line=dict(color='lightgrey', width=1)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_1"], name = 'Random Forest Model', line=dict(color='lightsteelblue', width=1)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_2"], name = 'XGBoost Model', line=dict(color='lightslategrey', width=1)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Ensemble"], name = 'Ensemble 2 (weighted)', line=dict(color='cadetblue', width=1)))
    fig.update_layout(title="Comparison of Forecasts by Model Type ", showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.99))
    fig.update_xaxes(nticks=10) 
    #fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    #fig.update_xaxes(minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000,), ticklabelmode="period", tickformat="%H:%M%p")
    #fig.update_yaxes(tickformat=",.0f")
    return fig


def show_model_evaluation(grid_generation, generation_type, forecast):
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_generation["date"], y=grid_generation[generation_type], name = format_generation_type_text(generation_type), line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Forecast_Ensemble"], name='Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="Model Training Performance", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_layout(hovermode="x unified")
    #fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig


def show_all_time_generation(grid_generation_all):
    # Create a plot of hourly forecast for today vs recorded wind power
    print(grid_generation_all[['month', 'wind(offshore)', 'wind(onshore)', 'solar', 'hydro']])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_generation_all["month"], y=grid_generation_all["wind(offshore)"], name = 'Wind (offshore)', line=dict(color='seagreen', width=2)))
    fig.add_trace(go.Scatter(x=grid_generation_all["month"], y=grid_generation_all["wind(onshore)"], name = 'Wind (onshore)', line=dict(color='lawngreen', width=2)))
    fig.add_trace(go.Scatter(x=grid_generation_all["month"], y=grid_generation_all["solar"], name = 'Solar', line=dict(color='yellow', width=2)))
    fig.add_trace(go.Scatter(x=grid_generation_all["month"], y=grid_generation_all["hydro"], name = 'Hydro', line=dict(color='mediumaquamarine', width=2)))
    fig.update_layout(title="UK Renewable Power Generation (MW) Monthly<br><sub>Avg daily generation</sub>", showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_xaxes(nticks=10) 
    return fig
    

def show_all_time_generation_bar(grid_generation_all):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grid_generation_all["year"], y=grid_generation_all["wind(offshore)"], name="Wind(offshore)", marker_color='seagreen'))
    fig.add_trace(go.Bar(x=grid_generation_all["year"], y=grid_generation_all["wind(onshore)"], name="Wind(onshore)", marker_color='lawngreen'))
    fig.add_trace(go.Bar(x=grid_generation_all["year"], y=grid_generation_all["solar"], name="Solar", marker_color='yellow'))
    fig.add_trace(go.Bar(x=grid_generation_all["year"], y=grid_generation_all["hydro"], name="Hydro", marker_color='mediumaquamarine'))
    fig.update_layout(title="UK Renewable Power Generation (MW) Yearly<br><sub>Avg daily generation</sub>", showlegend=True, barmode='stack')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_xaxes(nticks=10) 
    return fig


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
    fig.update_layout(title_text="Offshore Wind Farm Completions (MW Capacity)")
    fig.update_layout(width=500, height=400)
    return fig