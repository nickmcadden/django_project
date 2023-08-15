from django.shortcuts import render
from django_project import models
from django_project import data
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
    grid_generation = data.read_power_generation_data()#.groupby('date').mean().reset_index()
    grid_generation = grid_generation[grid_generation['date']>=pd.Timestamp.today().floor('D')]
    grid_generation['hour'] = grid_generation['date'] + pd.to_timedelta(grid_generation['period']/2-1, unit='h')    
    print(grid_generation)
    
    #grid_generation_today = grid_generation[grid_generation['date']>=pd.Timestamp.today().floor('D')]
    
    # Show the prediction for today
    forecast_today = forecast[forecast['Date']>=pd.Timestamp.today().floor('D')]
    forecast_today['Hour'] = forecast['Date'] + pd.to_timedelta(forecast['Hour']-1, unit='h')
    forecast_html = forecast_today.drop(['Date'], axis=1).to_html(classes='data-table', index=False, float_format = '{:,.0f}'.format)
    
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_training["Date"], y=daily_training["Wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=daily_forecast["Date"], y=daily_forecast["Forecast_Ensemble"], name='Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="Model Training Performance", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    
    # Convert the figure to an HTML div string
    daily_model_training = pio.to_html(fig, full_html=False, default_width='600px')
    
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
    #fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = np.arange(1,25).tolist()))
    
    today_wind_forecast = pio.to_html(fig, full_html=False, default_width='600px')
    
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_generation["hour"], y=grid_generation["wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Ensemble"], name = 'Ensemble Forecast', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="UK Wind Power Forecast (MW) " + pd.Timestamp.today().strftime("%A %d %B"), showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_xaxes(minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000,), ticklabelmode="period", tickformat="%H:%M%p")
    fig.update_yaxes(tickformat=",.0f")
    # Convert the figure to an HTML div string
    today_wind_actual = pio.to_html(fig, full_html=False, default_width='600px')
    
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_training["Date"], y=daily_training["Wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=daily_forecast["Date"], y=daily_forecast["Forecast_Ensemble"], name='Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(title="Model Training Performance", legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    
    # Convert the figure to an HTML div string
    daily_forecast_chart = pio.to_html(fig, full_html=False, default_width='600px')
    
    return render(request, 'index.html', {'forecast_data': forecast_html, 
                                          'model_training': daily_model_training,
                                          'today_wind_forecast': today_wind_forecast,
                                          'today_wind_actual': today_wind_actual})
