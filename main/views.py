from django.shortcuts import render
from django_project import models
from django_project import data
from django_project import visualisations
from datetime import date as dt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np


def index_view(request):
    # Get the latest forecast data
    forecastdata = data.read_forecast_data_old('2023-05-08')
    training_prediction, forecast = models.create_forecast(forecastdata)
    
    # Create a daily view of the forecast and training data
    daily_forecast = forecast.groupby('Date').mean().reset_index()
    daily_training = training_prediction.groupby('Date').mean().reset_index()
    
    # Get the grid generation data
    grid_generation = data.read_power_generation_data()
    
    # Filter data to today to add to the forecast chart to show actual vs forecast
    grid_generation_today = grid_generation[grid_generation['date']>=pd.Timestamp.today().floor('D')]
    grid_generation_today['hour'] = grid_generation_today['date'] + pd.to_timedelta((grid_generation_today['period']-1)/2, unit='h')
    
    # Show the prediction for today
    forecast_today = forecast[forecast['Date']>=pd.Timestamp.today().floor('D')]
    forecast_today['Hour'] = forecast['Date'] + pd.to_timedelta(forecast['Hour']-1, unit='h')
    forecast_html = forecast_today.drop(['Date'], axis=1).to_html(classes='data-table', index=False, float_format = '{:,.0f}'.format)
    
    # Create a chart showing the latest power generation
    fig_sunburst = visualisations.show_power_generation_chart(grid_generation, 'latest')
    latest_generation = fig_sunburst.to_html(full_html=False, include_plotlyjs=True)
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(daily_training, daily_forecast)
    daily_model_training = pio.to_html(fig_model_evaluation, full_html=False, default_width='600px')
    
    # Convert the figure to an HTML div string
    fig_todays_forecast = visualisations.show_todays_forecast(forecast_today)
    today_wind_forecast = pio.to_html(fig_todays_forecast, full_html=False, default_width='600px')
    
    # Convert the figure to an HTML div string
    fig_forecast_vs_actual = visualisations.show_forecast_vs_actual(grid_generation_today, forecast_today)
    today_wind_actual = pio.to_html(fig_forecast_vs_actual, full_html=False, default_width='600px')

    # Get a wind farm map
    folium_header, map_html, map_script = visualisations.uk_wind_power_map()
    
    # Send all the generated html to the index.html template
    return render(request, 'index.html', {'forecast_data': forecast_html,
                                          'latest_generation': latest_generation,
                                          'model_training': daily_model_training,
                                          'today_wind_forecast': today_wind_forecast,
                                          'today_wind_actual': today_wind_actual,
                                          'folium_header' : folium_header,
                                          'map_html': map_html,
                                          'map_script': map_script})

