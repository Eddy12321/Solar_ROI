
# # Predicting Power Using the NSRDB

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import __version__ as sklearn_version
from scipy.spatial import cKDTree
import requests
import json
from requests.structures import CaseInsensitiveDict

GEOAPI_KEY = st.secrets['GEOAPI_KEY']
NSRDBAPI_KEY = "gIgh6128lU37WJnlZyJuSOHICrGTn1C59Z8tBnD8"
newAddress = "f"

st.title("Predicting Solar Power Using the NSRDB")
st.markdown("This program will allow you to determine a timeline for paying off your solar installation. We begin by predicting the amount of solar power that can be generated at \
     the entered location. We then use the expected installation cost and the average electricity cost in your region to let you know when you can expect to start turning a profit!")

def main():
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

    x_train = pd.read_csv('./data/X_train2.csv')
    y_train = pd.read_csv('./data/Y_train2.csv')
    Electricity_cost = pd.read_csv('./data/StatesElectricity.csv')
    url1 = "https://api.geoapify.com/v1/geocode/search?text=" + Address + "%20United%20States&apiKey=" + GEOAPI_KEY

    # Powered by Geoapify https://www.geoapify.com
    headers = CaseInsensitiveDict()
    headers['Accept'] = "application/json"
    resp1 = requests.get(url1, headers = headers)
    features1 = resp1.json()['features']
    properties1 = features1[0]['properties']
    lat = properties1['lat']
    lon = properties1['lon']
    state1 = properties1['state']

    url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes)

    Locationdf = pd.read_csv(url)

    Locationdf = Locationdf.iloc[:,:12]
    Locationdf.columns = Locationdf.loc[1]
    Locationdf.drop([0,1], inplace = True)
    Locationdf.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)

    y_train.drop(columns = ['Unnamed: 0'], inplace = True)
    x_train.drop(columns = ['Unnamed: 0'], inplace = True)

    model = make_pipeline(MinMaxScaler(), PCA(n_components = 9), GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 77, max_depth = 6))

    Electricity_cost.set_index('State', inplace = True)
    Electricity_cost.drop(columns = 'Unnamed: 0', inplace = True)

    column_mapper2 = {'Temperature': 'TempOut',
                    'Pressure': 'Bar',
                    'Relative Humidity': 'OutHum',
                    'Wind Speed': 'WindSpeed',
                    'Dew Point': 'DewPt',
                    'GHI': 'SolarRad',
                    'Precipitable Water': 'Rain'}
    Locationdf.rename(columns = column_mapper2, inplace=True)

    for col in Locationdf.columns:
        Locationdf[col] = Locationdf[col].astype(float)

    Locationdf['Rain']  = Locationdf['Rain'] / 0.25
    Locationdf['SolarEnergy'] = Locationdf['SolarRad'] * 0.5 / 11.622
    Locationdf['Temp_Pressure_ratio'] = Locationdf['TempOut'] / Locationdf['Bar']

    Locationdf.replace(to_replace = {np.Inf: np.NAN}, inplace = True)
    Locationdf.dropna(inplace = True)

    model.fit(x_train, y_train)

    Locationdf['Power'] = model.predict(Locationdf)
    Locationdf['Power'] = Locationdf['Power'] * (InstallationSize / 0.6)

    StateElectricityCost1 = Electricity_cost.loc[state1]['Electricity Prices']

    Locationdf['Money_saved'] = Locationdf['Power'] * (StateElectricityCost1 / 2)

    dollars_saved1 = Locationdf['Money_saved'].sum() / 100

    st.write('The amount that you would save per year at your location is $' + str(dollars_saved1))
    TimeTillPayed = InstallationCost / dollars_saved1
    st.write('By the estimation, you would be aple to pay off your solar installation in ' + str(TimeTillPayed) + ' years!')

    Date = pd.date_range(start = "01/01/22 00:00:00", periods = Locationdf.shape[0], freq = '30T')

    st.write(Date)
    st.write(Locationdf.shape[0])

Address = st.text_input('Please enter your address as it appears on google, except with a space on both sides of the commas. No need to enter the country.', value = "1600 Pennsylvania Avenue , Washington DC , 20500")

tempAddress = Address.rsplit(" ")
Address = ""
for idx, word in enumerate(tempAddress):
    if idx == 0:
        Address += word
    else:
        if word == ',':
            Address +=  '%2C'
        else:
            Address += '%20' + word

InstallationCost = st.number_input('Please enter the expected cost of your solar panel installation in dollars.', value = 2400, min_value = 1)

InstallationSize = st.number_input('Please enter the size of your installation in kW.', value = 0.6, min_value = 0.05)

main()