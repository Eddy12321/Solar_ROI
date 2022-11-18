
# # Predicting Power Using the NSRDB

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
from sklearn import __version__ as sklearn_version
from scipy.spatial import cKDTree
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


GEOAPI_KEY = st.secrets['GEOAPI_KEY']
NSRDBAPI_KEY = "gIgh6128lU37WJnlZyJuSOHICrGTn1C59Z8tBnD8"
Address = "1600%20Pennsylvania%20Avenue%20NW%2C%20Washington%2C%20DC%2020500%"
InstallationCost = 2400
year = 2020
attributes = 'air_temperature,relative_humidity,dew_point,wind_speed,surface_pressure,total_precipitable_water,ghi'
leap_year = 'false'
interval = '30'
utc = 'false'
your_name = 'None+None'
reason_for_use = 'beta+testing'
your_affiliation = 'Springboard'
your_email = 'edwardjs43@gmail.com'
mailing_list = 'false'

x_train = pd.read_csv('../data/X_train2.csv')
y_train = pd.read_csv('../data/Y_train2.csv')
Electricity_cost = pd.read_csv('../data/StatesElectricity.csv')
Attributes = pd.read_csv('../data/Attributes.csv')
url1 = "https://api.geoapify.com/v1/geocode/search?text=" + Address + "%20United%20States&apiKey=" + GEOAPI_KEY

# Powered by Geoapify https://www.geoapify.com
headers = CaseInsensitiveDict()
headers['Accept'] = "application/json"
resp1 = requests.get(url1, headers = headers)
features1 = resp1.json()['features']
properties1 = features1[1]['properties']
lat = properties1['lat']
lon = properties1['lon']
state1 = properties1['state']

url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes)

Locationdf = pd.read_csv(url)

y_train.drop(columns = ['Unnamed: 0'],inplace = True)
x_train.drop(columns = ['Unnamed: 0'],inplace = True)

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
''''
Locationdf1 = getLocationData(lat1, lon1, cols = columns)
for col in columns:
    scale = float(Attributes.loc['scale_factor', col])
    Locationdf1[col] = Locationdf1[col] / scale
'''
column_mapper = {'air_temperature': 'TempOut',
                'surface_pressure': 'Bar',
                'relative_humidity': 'OutHum',
                'wind_speed': 'WindSpeed',
                'dew_point': 'DewPt',
                'ghi': 'SolarRad',
                'total_precipitable_water': 'Rain'}
Locationdf.rename(columns = column_mapper, inplace=True)

Locationdf['Bar'] = Locationdf['Bar']
Locationdf['Rain'] = Locationdf['Rain'] / 0.25
Locationdf['SolarEnergy'] = Locationdf['SolarRad'] * 0.5 / 11.622
Locationdf['Temp_Pressure_ratio'] = Locationdf['TempOut'] / Locationdf1['Bar']

Locationdf.replace(to_replace = {np.Inf: np.NAN}, inplace = True)
Locationdf.dropna(inplace = True)

model.fit(x_train, y_train)

Locationdf['Power'] = model.predict(Locationdf)

StateElectricityCost1 = Electricity_cost.loc[state1]['Electricity Prices']

Locationdf['Money_saved'] = Locationdf['Power'] * (StateElectricityCost1 / 2)

dollars_saved1 = Locationdf['Money_saved'].sum() / 100

print('The amount that you would save per year at your location is $' + str(dollars_saved1))
TimeTillPayed = InstallationCost / dollars_saved1
print('By the estimation, you would be aple to pay off your solar installation in ' + str(TimeTillPayed) + ' years!')