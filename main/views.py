from django.shortcuts import render
from django_project import models
from django_project import data
from django_project import visualisations
from django.conf import settings
from datetime import date as dt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np


def format_forecast_data(forecast_data):
    forecast = forecast_data.copy()[forecast_data['Date']>=pd.Timestamp.today().floor('D')]
    forecast['Hour'] = forecast['Date'] + pd.to_timedelta(forecast['Hour']-1, unit='h')
    return forecast
    

def index_view(request):
    # Get the latest forecast data
    forecast_wind_offshore = models.create_forecast('wind(offshore)')
    forecast_wind_onshore = models.create_forecast('wind(onshore)')
    forecast_solar = models.create_forecast('solar')
    
    # Get the grid generation data
    grid_generation = data.read_power_generation_data()

    # Filter data to today to add to the forecast chart to show actual vs forecast
    grid_generation_today = grid_generation.copy()[grid_generation['date']>=pd.Timestamp.today().floor('D')]
    grid_generation_today['hour'] = grid_generation_today['date'] + pd.to_timedelta((grid_generation_today['period']-1)/2, unit='h')
    
    #forecast_html = forecast_today.drop(['Date'], axis=1).to_html(classes='data-table', index=False, float_format = '{:,.0f}'.format)
    
    # Show the offshore wind forecasts and model comparison
    fig_forecast_vs_actual = visualisations.show_forecast_vs_actual(grid_generation_today, 'wind(offshore)', format_forecast_data(forecast_wind_offshore))
    wind_offshore_forecast_actual = pio.to_html(fig_forecast_vs_actual, full_html=False, default_width='1200px')
    
    # Show the onshore wind forecasts abd model comparison
    fig_forecast_vs_actual = visualisations.show_forecast_vs_actual(grid_generation_today, 'wind(onshore)', format_forecast_data(forecast_wind_onshore))
    wind_onshore_forecast_actual = pio.to_html(fig_forecast_vs_actual, full_html=False, default_width='1200px')
    
    # Show the solar forecasts and model comparison
    fig_forecast_vs_actual = visualisations.show_forecast_vs_actual(grid_generation_today, 'solar', format_forecast_data(forecast_solar))
    solar_forecast_actual = pio.to_html(fig_forecast_vs_actual, full_html=False, default_width='1200px')
    
    all_page_content = render(request, 'index.html', {'wind_offshore_forecast_actual': wind_offshore_forecast_actual,
                                        'wind_onshore_forecast_actual': wind_onshore_forecast_actual,
                                        'solar_forecast_actual': solar_forecast_actual})
    
    # Send all the generated html to the index.html template
    return all_page_content


def format_model_data(forecast_data):
    forecast = forecast_data.copy().groupby('Date').mean().reset_index()
    return forecast


def model_view(request):
    # Get the latest forecast data
    forecast_wind_offshore = models.create_forecast('wind(offshore)')
    forecast_wind_onshore = models.create_forecast('wind(onshore)')
    forecast_solar = models.create_forecast('solar')
    
    # Find the earliest date in the data set to filter the generation data by
    start_date = forecast_wind_offshore['Date'].min()

    # Get the grid generation data
    grid_generation = data.read_power_generation_data()
    # Filter to the time period for the forecast data (which also contains a historical forecast)
    grid_generation = grid_generation.copy()[grid_generation['date']>=start_date]
    # Create a new data set at the daily level
    grid_generation = grid_generation.copy().groupby('date').mean().reset_index()
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation, 'wind(offshore)', format_model_data(forecast_wind_offshore))
    wind_offshore_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation, 'wind(onshore)', format_model_data(forecast_wind_onshore))
    wind_onshore_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation, 'solar', format_model_data(forecast_solar))
    solar_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    all_page_content = render(request, 'models.html', {'wind_offshore_model_evaluation': wind_offshore_model_evaluation,
                                                        'wind_onshore_model_evaluation': wind_onshore_model_evaluation,
                                                        'solar_model_evaluation': solar_model_evaluation})
    
    # Send all the generated html to the template
    return all_page_content
    

def map_view(request):
    # Get UK wind farm data
    fig_uk_wind_farms = visualisations.show_uk_wind_farms()
    uk_wind_farm_chart = pio.to_html(fig_uk_wind_farms, full_html=False, default_width='600px')

    # Get a wind farm map
    folium_header, map_html, map_script = visualisations.uk_wind_power_map()
    
    # Get the grid generation data
    grid_generation = data.read_power_generation_data()
    
    # Get all the grid generation data at a daily level
    grid_generation['month'] = grid_generation['date'].to_numpy().astype('datetime64[M]')
    grid_generation['year'] = grid_generation['date'].dt.year  
    
    print(grid_generation[['date', 'month', 'year']])
    grid_generation_monthly = grid_generation.copy().groupby('month').mean().reset_index()
    grid_generation_yearly = grid_generation.copy().groupby('year').mean().reset_index()
    
    # Convert the figure to an HTML div string
    fig_all_time_generation = visualisations.show_all_time_generation(grid_generation_monthly)
    all_time_generation = pio.to_html(fig_all_time_generation, full_html=False, default_width='600px')
    
    # Convert the figure to an HTML div string
    fig_all_time_generation_bar = visualisations.show_all_time_generation_bar(grid_generation_yearly)
    all_time_generation_bar = pio.to_html(fig_all_time_generation_bar, full_html=False, default_width='600px')
    
    # Create a chart showing the latest power generation
    # fig_sunburst = visualisations.show_power_chart(grid_generation, 'latest')
    # latest_generation = fig_sunburst.to_html(full_html=False, include_plotlyjs=True)
    
    all_page_content = render(request, 'map.html', {'uk_wind_farms': uk_wind_farm_chart,
                                       'map_html': map_html,
                                       'map_script': map_script,
                                       'all_time_generation': all_time_generation,
                                       'all_time_generation_bar': all_time_generation_bar})
    
    # Send all the generated html to the index.html template
    return all_page_content


def test_view(request):
    forecast_wind_offshore = models.create_forecast('wind(offshore)')
    forecast_wind_onshore = models.create_forecast('wind(onshore)')
    #forecast_solar = models.create_forecast('solar')
    return all_page_content

'''
To Do:
Random Bloomsbury group generator
Random Pyjama party generator
'''
                                          