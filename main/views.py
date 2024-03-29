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
    grid_generation_daily = grid_generation.copy().groupby('date').mean().reset_index()
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation_daily, 'wind(offshore)', format_model_data(forecast_wind_offshore))
    wind_offshore_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation_daily, 'wind(onshore)', format_model_data(forecast_wind_onshore))
    wind_onshore_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    # Convert the figure to an HTML div string
    fig_model_evaluation = visualisations.show_model_evaluation(grid_generation_daily, 'solar', format_model_data(forecast_solar))
    solar_model_evaluation = pio.to_html(fig_model_evaluation, full_html=False, default_width='1200px')
    
    # Show an evaluation by time lag from when the forecast was created
    fig_model_evaluation_time_lag = visualisations.evaluate_forecast_timelag(grid_generation, ['wind(offshore)', 'wind(onshore)', 'solar'])
    time_lag_evaluation = pio.to_html(fig_model_evaluation_time_lag, full_html=False, default_width='1200px')

    all_page_content = render(request, 'models.html', {'wind_offshore_model_evaluation': wind_offshore_model_evaluation,
                                                        'wind_onshore_model_evaluation': wind_onshore_model_evaluation,
                                                        'solar_model_evaluation': solar_model_evaluation,
                                                        'time_lag_evaluation': time_lag_evaluation})
    
    # Send all the generated html to the template
    return all_page_content

# To Do (apply a colour change or transparency affect to this year's bar to show it's a projection not actual to date)
def add_current_year_projection(grid_generation):
    # Remove last years data between equivalent date (today year year ago) and end of that year
    today_last_year = pd.Timestamp.today() - pd.DateOffset(years=1)
    grid_generation_temp = grid_generation.copy()[grid_generation['date'] < today_last_year]
    grid_generation_temp = grid_generation_temp.append(grid_generation.copy()[(grid_generation['date'] >= '2024-01-01') & (grid_generation['date'] < pd.Timestamp.today())])
    
    # Aggregate yearly
    grid_generation_yearly = grid_generation_temp.copy().groupby('year').mean().reset_index()
    
    # create multiplier for the projection (this year's yearly aggregate / last years)
    grid_generation = grid_generation.set_index(['date'])
    this_year = grid_generation_yearly.copy()[grid_generation_yearly['year']==pd.Timestamp.today().year]
    last_year = grid_generation_yearly.copy()[grid_generation_yearly['year']==pd.Timestamp.today().year-1]
    
    multipliers = this_year.to_numpy() / last_year.to_numpy()
    
    # Aggregate yearly with all data
    grid_generation_yearly = grid_generation[grid_generation['year']<=pd.Timestamp.today().year].copy().groupby('year').mean().reset_index()
    # Get last year's totals to base the projection on
    last_year = grid_generation_yearly.copy()[grid_generation_yearly['year']==pd.Timestamp.today().year-1]
    # Apply new projection
    new_projection = multipliers *  last_year.to_numpy()
    
    # Print all the multiplieers and new projections
    print(pd.DataFrame({'columns' : grid_generation_yearly.columns, 'multipliers': multipliers.squeeze(), 'last year': last_year.to_numpy().squeeze(), 'new_projection': new_projection.squeeze()}))
    
    grid_generation_yearly.set_index('year')
    
    # Replace the data for current year with last year's data * multiplier for each type
    for i in range(grid_generation_yearly.shape[1]):
        grid_generation_yearly.iat[9,i] = new_projection.squeeze()[i]
    
    print('grid_generation_yearly', grid_generation_yearly[['wind(offshore)', 'wind(onshore)', 'solar']])
    
    return grid_generation_yearly


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
    
    # Create a monthly data set
    grid_generation_monthly = grid_generation.copy().groupby('month').mean().reset_index()
    
    # Create a yearly data set
    grid_generation_yearly = add_current_year_projection(grid_generation)
    
    # Convert the figure to an HTML div string
    fig_all_time_generation = visualisations.show_all_time_generation(grid_generation_monthly)
    all_time_generation = pio.to_html(fig_all_time_generation, full_html=False, default_width='600px')
    
    # Convert the figure to an HTML div string
    fig_all_time_generation_bar = visualisations.show_all_time_generation_bar(grid_generation_yearly)
    all_time_generation_bar = pio.to_html(fig_all_time_generation_bar, full_html=False, default_width='600px')
    
    # Convert the figure to an HTML div string
    fig_all_time_carbon_bar = visualisations.show_all_time_carbon_bar(grid_generation_yearly)
    all_time_carbon_bar = pio.to_html(fig_all_time_carbon_bar, full_html=False, default_width='600px')
    
    # Create a chart showing the latest power generation
    # fig_sunburst = visualisations.show_power_chart(grid_generation, 'latest')
    # latest_generation = fig_sunburst.to_html(full_html=False, include_plotlyjs=True)
    
    all_page_content = render(request, 'map.html', {'uk_wind_farms': uk_wind_farm_chart,
                                       'map_html': map_html,
                                       'map_script': map_script,
                                       'all_time_generation': all_time_generation,
                                       'all_time_generation_bar': all_time_generation_bar,
                                       'all_time_carbon_bar': all_time_carbon_bar})
    
    # Send all the generated html to the index.html template
    return all_page_content


def test_view(request):
    generation_data = data.read_power_generation_data()
    generation_data['demand'] = generation_data.sum(axis=1, numeric_only=True)
    generation_data = generation_data.copy()[generation_data['date']>='2022-09-01']
    generation_data.reset_index(inplace=True)
    
    # Show baseline scenario (current capacity)
    # Scenarios are created by passing in a dictionary in the form {'renewable_source1': capacity_multiplier, 'renewable_source2':capacity_multiplier}
    
    plans = []
    plans.append({'plan': 'current capacity', 'storage_capacity_GWh': 0, 'capacity':{}})
    plans.append({'plan': 'capacity boost', 'storage_capacity_GWh': 0, 'capacity': {'solar': 2, 'wind(offshore)': 2}})
    plans.append({'plan': 'capacity boost + 180GW', 'storage_capacity_GWh': 180, 'capacity': {'solar': 2, 'wind(offshore)': 2.5}})
    plans.append({'plan': 'capacity boost + 320GW', 'storage_capacity_GWh': 360, 'capacity': {'solar': 2, 'wind(offshore)': 2.5}}) 
    plans.append({'plan': 'capacity boost + 540GW', 'storage_capacity_GWh': 540, 'capacity': {'solar': 2, 'wind(offshore)': 2.5, 'nuclear':2}})
    plans.append({'plan': 'capacity boost + 720GW', 'storage_capacity_GWh': 720, 'capacity': {'solar': 2, 'hydro':2, 'wind(offshore)': 2.5, 'nuclear':2}})
    
    plan_data = []
    for i in range(len(plans)):
        plan_data.append(data.create_scenerio_data(generation_data.copy(), plans[i], 'all'))
    
    table_data = pd.DataFrame(columns=['plan', 'storage_capacity_GWh', 'wind(offshore)', 'wind(onshore)', 'solar', 'hydro', 'nuclear', 'biomass'])
    for i in range(len(plans)):
        table_data = table_data.append(plans[i], ignore_index=True)
    
    for i, capacity in enumerate(table_data['capacity'].values):
        for generation_type, capacity_multiplier in capacity.items():
            table_data.iloc[i, table_data.columns.get_loc(generation_type)] = capacity_multiplier
    
    table_data.fillna(1, inplace=True)
    table_data.drop(['capacity'], axis=1, inplace=True)
    
    plan_overview = table_data.to_html(classes='data-table', index=False, float_format = '{:,.0f}'.format)
    
    fig_capacity_projections = visualisations.show_capacity_projection(plans, plan_data)
    capacity_projections = pio.to_html(fig_capacity_projections, full_html=False, default_width='1200px', default_height='600px')
    
    all_page_content = render(request, 'test.html', {'capacity_projections': capacity_projections,
                                                     'plan_overview': plan_overview})
    
    return all_page_content

'''
To Do:
Random Bloomsbury group generator
Random Pyjama party generator
'''
                                          