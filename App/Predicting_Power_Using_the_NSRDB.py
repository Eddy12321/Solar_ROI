#!/usr/bin/env/python3
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    attributes1 = 'air_temperature,alpha,aod,asymmetry,cld_opd_dcomp,cld_reff_dcomp'
    #attributes2 = 'clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_press_acha,cloud_type,dew_point'
    #attributes3 = 'dhi,dni,ghi,ozone,relative_humidity,solar_zenith_angle,ssa,surface_albedo'
    #attributes4 = 'surface_pressure,total_precipitable_water,wind_direction,wind_speed'
    leap_year = 'false'
    interval = '30'
    utc = 'false'
    your_name = 'None+None'
    reason_for_use = 'beta+testing'
    your_affiliation = 'Springboard'
    your_email = 'edwardjs43@gmail.com'
    mailing_list = 'false'

    x_train = pd.read_csv('./data/X_train_New.csv')
    y_train = pd.read_csv('./data/Y_train_New.csv')
    Electricity_cost = pd.read_csv('./data/StatesElectricity.csv')
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

    url2 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes1)
    #url3 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes2)
    #url4 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes3)
    #url5 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes4)
    
    Locationdf1 = pd.read_csv(url2)
    #Locationdf2 = pd.read_csv(url3)
    #Locationdf3 = pd.read_csv(url4)
    #Locationdf4 = pd.read_csv(url5)

    Locationdf1 = Locationdf1.iloc[:,:12]
    Locationdf1.columns = Locationdf1.loc[1]
    Locationdf1.drop([0,1], inplace = True)
    Locationdf1.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)
    Locationdf2 = Locationdf2.iloc[:,:12]
    Locationdf2.columns = Locationdf2.loc[1]
    Locationdf2.drop([0,1], inplace = True)
    Locationdf2.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)
    Locationdf3 = Locationdf3.iloc[:,:12]
    Locationdf3.columns = Locationdf3.loc[1]
    Locationdf3.drop([0,1], inplace = True)
    Locationdf3.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)
    Locationdf4 = Locationdf4.iloc[:,:12]
    Locationdf4.columns = Locationdf4.loc[1]
    Locationdf4.drop([0,1], inplace = True)
    Locationdf4.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)
    Locationdf = Locationdf1.merge(Locationdf2, left_index = True, right_index = True).merge(Locationdf3, left_index = True, right_index = True).merge(Locationdf4, left_index = True, right_index = True)

    y_train.drop(columns = ['Unnamed: 0'], inplace = True)
    x_train.drop(columns = ['Unnamed: 0'], inplace = True)

    model = make_pipeline(MinMaxScaler(), PCA(n_components = 16), RandomForestRegressor(n_estimators = 50, max_depth = 16))

    Electricity_cost.set_index('State', inplace = True)
    Electricity_cost.drop(columns = 'Unnamed: 0', inplace = True)

    for col in Locationdf.columns:
        Locationdf[col] = Locationdf[col].astype(float)

    Locationdf.dropna(inplace = True)

    model.fit(x_train, y_train)

    Locationdf['Power'] = model.predict(Locationdf) * (InstallationSize / 0.6)

    StateElectricityCost1 = Electricity_cost.loc[state1]['Electricity Prices']

    Locationdf['Money_saved'] = Locationdf['Power'] * (StateElectricityCost1 / 2)

    dollars_saved1 = Locationdf['Money_saved'].sum() / 100

    st.write('The amount that you would save per year at your location is $' + str(dollars_saved1))
    TimeTillPayed = InstallationCost / dollars_saved1
    st.write('By the estimation, you would be aple to pay off your solar installation in ' + str(TimeTillPayed) + ' years!')

    Locationdf['Date'] = pd.date_range(start = "01/01/22 00:00:00", periods = Locationdf.shape[0], freq = '30T')
    Locationdf['Month'] = Locationdf['Date'].agg(func = lambda x: x.month)

    fig, ax = plt.subplots(figsize = (6, 4))
    ax.hist(Locationdf['Power'])
    ax.set(xlabel='Power Generated', ylabel='Count', title = 'Histogram of Power Generated')

    st.pyplot(fig = fig)

    def scatterplots(columns, ncol=None, figsize=(15, 8)):
        if ncol is None:
            ncol = len(columns)
        nrow = int(np.ceil(len(columns) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        fig.subplots_adjust(wspace=0.5, hspace=0.6)
        for i, col in enumerate(columns):
            ax = axes.flatten()[i]
            ax.plot(Locationdf['Month'].unique(), Locationdf.groupby('Month')[col].mean())
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            ax.set(xlabel='Month', ylabel='Average '+col+' by month')
        nsubplots = nrow * ncol    
        for empty in range(i+1, nsubplots):
            axes.flatten()[empty].set_visible(False)

    scatterplots(['Power', 'air_temperature', 'dhi', 'dew_point', 'relative_humidity', 'cloud_press_acha', 'surface_pressure'], figsize = (14, 4))

    st.pyplot(fig = fig)

Address = st.text_input('Please enter your address as it appears on google, except with a space on both sides of the commas. No need to enter the country.', value = "1600 Pennsylvania Avenue , Washington DC , 20500")

tempAddress = Address.rsplit(" ")
Address = ""
Address += tempAddress[0]
for idx, word in enumerate(tempAddress, start = 1):
    if word == ',':
        Address +=  '%2C'
    else:
        Address += '%20' + word

InstallationCost = st.number_input('Please enter the expected cost of your solar panel installation in dollars.', value = 2400, min_value = 1)

InstallationSize = st.number_input('Please enter the size of your installation in kW.', value = 0.6, min_value = 0.05)

main()