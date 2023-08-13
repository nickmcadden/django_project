from django.shortcuts import render
from django_project import models
from django_project import data
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
    
    # Show the prediction for today
    forecast_today = forecast[forecast['Date']>=pd.Timestamp.today().floor('D')]
    forecast_html = forecast_today.to_html(classes='data-table', index=False, float_format = '{:,.0f}'.format)
    
    # Create a plot of the daily forecast vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_training["Date"], y=daily_training["Wind"], name = 'Wind Power', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=daily_forecast["Date"], y=daily_forecast["Forecast_Ensemble"], name='Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    
    # Convert the figure to an HTML div string
    daily_forecast_chart = pio.to_html(fig, full_html=False, default_width='600px')
    
    # Create a plot of hourly forecast for today vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_today["Hour"], y=forecast_today["Forecast_Ensemble"], name = 'Forecast (MW)', line=dict(color='royalblue', width=2)))
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01))
    fig.update_layout(xaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(1,25).tolist()
        )
    )
    
    today_forecast_chart = pio.to_html(fig, full_html=False, default_width='600px')
    
    return render(request, 'index.html', {'forecast_data': forecast_html, 
                                          'daily_chart': daily_forecast_chart,
                                          'today_chart': today_forecast_chart})
