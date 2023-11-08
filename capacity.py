import os
import re
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

#################################################################################
#                                                                               #
#################################################################################

renewable_sources = ['wind(offshore)', 'wind(onshore)', 'solar', 'hydro']
renewable_sources_plus_battery = ['wind(offshore)', 'wind(onshore)', 'solar', 'hydro', 'battery_supplied']
carbon_sources = ['coal', 'ccgt']
carbon = ['carbon']
other_sources = ['nuclear', 'biomass']

# Set the available battery storage capacity
storage_capacity_MWh = 0
# The data set trades battery capacity by the half hour period and all power figures are in KWh
storage_capacity_KWhh = storage_capacity_MWh * 1000 * 2

def create_subheading(capacity):
    text = '<sub>Storage Capacity: ' +  str(storage_capacity_MWh) + ' MWh   '
    for column, capacity_multiplier in capacity.items():
        text = text + column + ': '+'x '+ str(capacity_multiplier)+'   '
    text = text + '</sub>'
    return text


def show_demand_vs_renewables(chart_data, capacity, group_by='month'):
    # Create a plot of hourly forecast for today vs recorded wind power
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["wind(offshore)_contribution"], name="Wind(offshore)", marker_color='seagreen'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["wind(onshore)_contribution"], name="Wind(onshore)", marker_color='lawngreen'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["solar_contribution"], name="Solar", marker_color='yellow'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["hydro_contribution"], name="Hydro", marker_color='mediumaquamarine'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["nuclear_contribution"], name="Nuclear", marker_color='ghostwhite'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["biomass_contribution"], name="Biomass", marker_color='dodgerblue'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["carbon_contribution"], name="Carbon", marker_color='lemonchiffon'))
    fig.add_trace(go.Bar(x=chart_data[group_by], y=chart_data["battery_supplied_contribution"], name="Battery", marker_color='blue'))
    fig.update_layout(title="UK Renewables % contribution to demand by month<br>"+create_subheading(capacity), showlegend=True, barmode='stack')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.99))
    fig.update_xaxes(nticks=15)
    return fig


def balance_non_carbon_generation(data, row_number, row_data, generation_type, excess):
    if excess > row_data[generation_type]:
        remaining_excess = excess - row_data[generation_type]
        data.iloc[row_number, data.columns.get_loc(generation_type)] = 0
    else:
        remaining_excess = 0
        data.iloc[row_number, data.columns.get_loc(generation_type)] = data.iloc[row_number, data.columns.get_loc(generation_type)] - excess
    return data, remaining_excess


def balance_carbon_generation(data, row_number, excess):
    data.iloc[row_number, data.columns.get_loc('carbon')] = np.abs(excess)
    remaining_excess = 0
    return data, remaining_excess


def create_scenerio_data(data, capacity, group_by='month'):
    # Calculate the new capacity
    for column, capacity_multiplier in capacity.items():
        data[column] = data[column] * capacity_multiplier
    # The new generation data will need clipping if it exceeds demand. 
    # And the extra production reclassified as battery generated
    # Iterate through by hour and clip the offshore wind generation where total_renewables exceeds demand
    # Keep log of: hours of total renewable demand coverage
    data['battery_supplied'] = 0
    data['wasted_KWhh'] = 0

    stored_KWhh = 0
    wasted_KWhh = []
    battery_supplied = []
    excess_generated = []
    for i, row in data.iterrows():
        excess = row[renewable_sources + other_sources].sum() - row['demand']
        if excess > 0:
            excess_generated.append(excess)
            battery_supplied.append(0)
            stored_KWhh += excess
            if stored_KWhh > storage_capacity_KWhh:
                wasted_KWhh.append(stored_KWhh - storage_capacity_KWhh)
                stored_KWhh = storage_capacity_KWhh
            else:
                wasted_KWhh.append(0)
        elif stored_KWhh > np.abs(excess):
            excess_generated.append(0)
            wasted_KWhh.append(0)
            battery_supplied.append(np.abs(excess))
            stored_KWhh = stored_KWhh - np.abs(excess)
        else:
            excess_generated.append(excess + stored_KWhh)
            wasted_KWhh.append(0)
            battery_supplied.append(np.abs(stored_KWhh))
            stored_KWhh = 0
    
    # Add all the hourly trade in excess production and release to the dataset
    data['battery_supplied'] = battery_supplied
    data['wasted_KWhh'] = wasted_KWhh
    data['excess_generated'] = excess_generated
    
    data.reset_index(inplace=True)
    
    # Default all carbon generation to 0 before calculating how much of this needs to be generated in the new scenario
    data['carbon'] = 0
    
    # Adjust the balance of carbon and non carbon sources based on the new renewable scenario 
    for i, row in data.iterrows():
        excess = row['excess_generated']
        if excess > 0:
            # balance non carbon sources in order of precedence
            for generation_type in ['nuclear', 'biomass', 'wind(offshore)', 'wind(onshore)']:
                data, excess = balance_non_carbon_generation(data, i, row, generation_type, excess)
                if excess == 0:
                    break
        else:
            data, excess = balance_carbon_generation(data, i, excess)
    
    #data['wind(offshore)'] = data['wind(offshore)'] - excess_generated
    data['renewables+battery'] = data[renewable_sources_plus_battery].sum(axis=1)
    # Create the new aggregation after the redistribution of any clipped data
    data['month'] = data['date'].to_numpy().astype('datetime64[M]')
    data['year'] = data['date'].dt.year
    
    if group_by == 'month':
        data = data.copy().groupby('month').mean().reset_index()
    if group_by == 'year':
        data = data.copy().groupby('year').mean().reset_index()
    
    # Calculate % contribution to demand for each source
    print(data.columns)
    for i in renewable_sources_plus_battery + other_sources + carbon:
        data[i+'_contribution'] = np.round(data[i] / data['demand'] * 100, 1)
    
    return data



# Show baseline scenario (current capacity)
# Scenarios are created by passing in a dictionary in the form {'renewable_source1': capacity_multiplier, 'renewable_source2':capacity_multiplier}
capacity = {'solar': 2, 'wind(offshore)': 2}
chart_data = create_scenerio_data(generation_data, capacity, 'month')
fig_demand_vs_renewables = show_demand_vs_renewables(chart_data, capacity, 'month')
fig_demand_vs_renewables.show()

'''
# Wind data can be modelled by sampling from the current available data.
# But there is a dependance between two consecutive periods. (Today's generation is more likely to be closer to yesterday's than a random day. This is an important consideration when modelling battery storage requirements)

wind(offshore)_data = np.array(generation_data.copy()[generation_data['date']>='2021-11-01']['wind(offshore)'].values)
wind(offshore)_data2 = np.append(wind(offshore)_data.copy()[0], wind(offshore)_data.copy()[:-1])
print(np.sum(np.abs(wind(offshore)_data - (wind(offshore)_data2)))/len(wind(offshore)_data))
wind(offshore)_data2 = np.random.choice(wind(offshore)_data, size=len(wind(offshore)_data), replace=False)
print(np.sum(np.abs(wind(offshore)_data - (wind(offshore)_data2)))/len(wind(offshore)_data))

# Create a histogram
plt.hist(data, bins=25, color='blue', alpha=0.2)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the Dataset')
plt.grid(True)

# Show the histogram
plt.show()

# Calculate summary statistics
mean = np.mean(data)
minimum = np.min(data)
maximum = np.max(data)

# Calculate the 5th and 95th percentiles
percentile_5th = np.percentile(data, 5)
percentile_95th = np.percentile(data, 95)

# Print summary statistics
print("Summary Statistics:")
print(f"Mean: {mean:.2f}")
print(f"Minimum: {minimum:.2f}")
print(f"Maximum: {maximum:.2f}")

# Print the percentiles
print("5th Percentile: {:.2f}".format(percentile_5th))
print("95th Percentile: {:.2f}".format(percentile_95th))

# Parameters for the negative binomial distribution
lambda_ = 4

# Generate a dataset with a left skew using the negative binomial distribution
# Generate a left-skewed dataset using the Poisson distribution
data = np.random.poisson(lambda_, size=1000)# * t

# Create a histogram to visualize the data
plt.hist(data, bins=25, color='blue', alpha=1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Left-Skewed Dataset')
plt.grid(True)
plt.show()
'''
