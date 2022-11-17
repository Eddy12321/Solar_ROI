
# # Predicting Power Using the NSRDB

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version
from scipy.spatial import cKDTree
import h5pyd
import requests
import json
from requests.structures import CaseInsensitiveDict

def getLocationData(lat, lon, cols = ['meta']):
    
    tree = cKDTree(nsrdb['coordinates'])
    dist, pos = tree.query(np.array([lat, lon]))
    
    df = pd.DataFrame(columns = cols)
    
    for col in cols:
        df[col] = nsrdb[col][:, pos]
    
    return df


API_KEY = st.secrets['GEOAPI_KEY']
Address = "1600%20Pennsylvania%20Avenue%20NW%2C%20Washington%2C%20DC%2020500%"
InstallationCost = 2400

nsrdb = h5pyd.File("/nrel/nsrdb/v3/nsrdb_2020.h5", 'r')
x_train = pd.read_csv('../data/X_train2.csv')
y_train = pd.read_csv('../data/Y_train2.csv')
Electricity_cost = pd.read_csv('../data/StatesElectricity.csv')
Attributes = pd.read_csv('../data/Attributes.csv')
url1 = "https://api.geoapify.com/v1/geocode/search?text=" + Address + "%20United%20States&apiKey=" + API_KEY

y_train.drop(columns = ['Unnamed: 0'],inplace = True)
x_train.drop(columns = ['Unnamed: 0'],inplace = True)

# Powered by Geoapify https://www.geoapify.com

headers = CaseInsensitiveDict()
headers['Accept'] = "application/json"
resp1 = requests.get(url1, headers = headers)

print(resp1.status_code)

features1 = resp1.json()['features']

properties1 = features1[1]['properties']
lat1 = properties1['lat']
lon1 = properties1['lon']
state1 = properties1['state']

expected_model_version = '1.0'
model_path = '../models/power_predictor.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if model.version != expected_model_version:
        print("Expected model version doesn't match version loaded")
    if model.sklearn_version != sklearn_version:
        print("Warning: model created under different sklearn version")
else:
    print("Expected model not found")

Electricity_cost.set_index('State', inplace = True)
Electricity_cost.drop(columns = 'Unnamed: 0', inplace = True)

Attributes.set_index('Unnamed: 0', inplace = True)

columns = ['air_temperature', 'relative_humidity', 'dew_point', 'wind_speed', 'surface_pressure', 'total_precipitable_water', 'ghi']

Locationdf1 = getLocationData(lat1, lon1, cols = columns)

for col in columns:
    scale = float(Attributes.loc['scale_factor', col])
    Locationdf1[col] = Locationdf1[col] / scale

column_mapper = {'air_temperature': 'TempOut',
                'surface_pressure': 'Bar',
                'relative_humidity': 'OutHum',
                'wind_speed': 'WindSpeed',
                'dew_point': 'DewPt',
                'ghi': 'SolarRad',
                'total_precipitable_water': 'Rain'}
Locationdf1.rename(columns = column_mapper, inplace=True)

Locationdf1['Bar'] = Locationdf1['Bar']
Locationdf1['Rain'] = Locationdf1['Rain'] / 0.25
Locationdf1['SolarEnergy'] = Locationdf1['SolarRad'] * 0.5 / 11.622
Locationdf1['Temp_Pressure_ratio'] = Locationdf1['TempOut'] / Locationdf1['Bar']

Locationdf1.replace(to_replace = {np.Inf: np.NAN}, inplace = True)
Locationdf1.dropna(inplace = True)

model.fit(x_train, y_train)

Locationdf1['Power'] = model.predict(Locationdf1)

StateElectricityCost1 = Electricity_cost.loc[state1]['Electricity Prices']

Locationdf1['Money_saved'] = Locationdf1['Power'] * (StateElectricityCost1 / 2)

dollars_saved1 = Locationdf1['Money_saved'].sum() / 100

print('The amount that you would save per year at your location is $' + str(dollars_saved1))
TimeTillPayed = InstallationCost / dollars_saved1
print('By the estimation, you would be aple to pay off your solar installation in ' + str(TimeTillPayed) + ' years!')