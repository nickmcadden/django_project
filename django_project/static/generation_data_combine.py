import os
import pandas as pd
import numpy as np

data = pd.read_csv('GenerationbyFuelType_20220701_to_present.csv')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)