import os
import pandas as pd
import numpy as np
import sqlite3
import datetime


def join_data_sets(location, base_file):
    f = open(os.path.join(location, base_file))
    data_columns = f.readline().replace('\n', '').split(',')
    print(data_columns)
    data = pd.read_csv(os.path.join(location, base_file), usecols=data_columns)
    print(os.listdir(location))
    for filename in os.listdir(location):
        f = os.path.join(location, filename)
        print(f)
        # checking if it is a file and isn't the base data
        if os.path.isfile(f) and filename != base_file and filename !='.DS_Store':
            if location == 'data/ntn':
                data_to_add = pd.read_csv(f, engine= 'python', skiprows=1, skipfooter=1, header=None)
                data_to_add.columns = data.columns
            if location == 'data/grid':
                data_to_add = pd.read_csv(f, usecols=data_columns)
            print(data.shape, data_to_add.shape)
            data = pd.concat([data, data_to_add], axis=0)
    return data


def test_for_missing_data_and_duplicates(data, date_column):
    for i, row in data.iterrows():
        d = row[date_column]
        if i==0:
            lastdate = row[date_column]
            continue
        else:
            datediff = (d - lastdate).days
            if datediff > 1:
                print(d, datediff-1, 'days missing')
        if row[1] < 48:
            print(d, 'only', row[1], 'periods')
        if row[1] > 48:
            print(d, 'duplicates (', row[1], ')periods')
        lastdate = d
      
def format_ntn_data(ntn_data):
    data_mapping_ntn_columns = {'startTimeOfHalfHrPeriod': 'date',
                    'settlementPeriod': 'period',
                    'coal': 'coal',
                    'ccgt': 'ccgt',
                    'ocgt': 'ocgt',
                    'nuclear': 'nuclear',
                    'oil': 'oil',
                    'wind': 'wind(offshore)',
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
                    'EMBEDDED_WIND_GENERATION':'wind(onshore)',
                    'EMBEDDED_SOLAR_GENERATION':'solar',
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


# Create the data set for the national transmission network
data = join_data_sets('data/ntn', 'GenerationbyFuelType_20220701_to_present.csv')
# Add a date, year , month and day column
data['startTimeOfHalfHrPeriod'] = data['startTimeOfHalfHrPeriod'].apply(lambda x: str(x)[:4] + str(x)[4:6] + str(x)[6:8]).astype('datetime64[ns]')
data = format_ntn_data(data)
data.to_csv('generation_half_hourly_ntn.csv', index=False)

# Create some test data to check if if there is anything missing
data_test1 = data.groupby(['date']).count().reset_index()
test_for_missing_data_and_duplicates(data_test1, 'date')

# Create the data set for the rest of the national grid supply
data = join_data_sets('data/grid', 'demanddata_2009.csv')
data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].astype('datetime64[ns]')
data = format_grid_data(data)
data.to_csv('generation_half_hourly_grid.csv', index=False)

# Create some test data to check if if there is anything missing
data_test2 = data.groupby(['date']).count().reset_index()
test_for_missing_data_and_duplicates(data_test2, 'date')

'''
# Connect to the database or create one if it doesn't exist
conn = sqlite3.connect('netzero.db')
cursor = conn.cursor()

# cursor.execute('
CREATE TABLE IF NOT EXISTS generation_half_hourly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP,
    period INTEGER,
    coal INTEGER,
    gas INTEGER,
    ocgt INTEGER,
    nuclear INTEGER,
    oil INTEGER,
    wind INTEGER,
    hydro INTEGER,
    pumped INTEGER,
    biomass INTEGER,
    other INTEGER,
    embedded_wind INTEGER,
    solar INTEGER,
    pumped_storage_pumping INTEGER,
    ifa INTEGER,
    ifa2 INTEGER,
    britned INTEGER,
    moyle INTEGER,
    ewic INTEGER,
    nemo INTEGER,
    nsl INTEGER,
    eleclink INTEGER
)
'')
print("Table created!")

import datetime
for i, row in data.iterrows():
    cursor.execute("INSERT INTO generation_half_hourly (date,period,coal,gas,ocgt,nuclear,oil,wind,hydro,pumped,biomass,other,embedded_wind,solar,pumped_storage_pumping,ifa,ifa2,britned,moyle,ewic,nemo,nsl,eleclink VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (row.date, row.period, row.coal, row.gas, row.ocgt, row.nuclear, row.oil, row.wind, row.hydro, row.pumped, row.biomass, row.other, row.embedded_wind, row.solar, row.pumped_storage_pumping, row.ifa, row.ifa2, row.britned, row.moyle,row.ewic, row.nemo, row.nsl, row.eleclink))

print("Data inserted!")

# Retrieve and print data from the 'users' table
cursor.execute("SELECT * FROM generation_half_hourly")
rows = cursor.fetchall()
print("\nData from 'generation_half_hourly' table:")
for row in rows:
    print(row)

# Update data in the 'users' table
# cursor.execute("UPDATE users SET age = ? WHERE name = ?", (26, "John Doe"))
# print("Data updated!")

# Commit the changes and close the connection
conn.commit()
conn.close()
'''