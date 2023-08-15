import os
import pandas as pd
import numpy as np
import pickle as pkl
import requests
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
    
a = read_power_generation_data()
