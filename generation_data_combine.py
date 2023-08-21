import os
import pandas as pd
import numpy as np

data = pd.read_csv('GenerationbyFuelType_20220701_to_present.csv')

print(data.shape)
for filename in os.listdir('data'):
    f = os.path.join('data', filename)
    # checking if it is a file and isn't the base data
    if os.path.isfile(f) and filename != 'GenerationbyFuelType_20220701_to_present.csv':
        print(f)
        data_to_add = pd.read_csv(f, engine= 'python', skiprows=1, skipfooter=1, header=None)
        data_to_add.columns = data.columns
        print(data.shape, data_to_add.shape)
        data = data.append(data_to_add)

print(data)

data['date'] = data['startTimeOfHalfHrPeriod'].apply(lambda x: str(x)[:4] + str(x)[4:6] + str(x)[6:8]).astype('datetime64[ns]')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

data_test1 = data.groupby(['year', 'month', 'day']).count().reset_index()
data_test1.to_csv('test_counts1.csv')

data_test2 = data.groupby(['year', 'month']).count()
data_test2 = round(data_test2 / 48, 1)
data_test2.to_csv('test_counts2.csv')

import datetime
for i, row in data_test1.iterrows():
    d = datetime.date(row['year'], row['month'], row['day'])
    if i==0:
        lastdate = d
        continue
    else:
        datediff = (d - lastdate).days
        if datediff > 1:
            print(d, datediff-1, 'days missing')
    if row['report'] < 48:
        print(d, 'only', row['report'], 'periods')
    if row['report'] > 48:
        print(d, 'duplicates (', row['report'], ')periods')
    lastdate = d
