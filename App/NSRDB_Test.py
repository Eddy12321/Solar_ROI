#!/usr/bin/env python3
# # Predicting Power Using the NSRDB
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import __version__ as sklearn_version
from scipy.spatial import cKDTree
import requests
import json
from requests.structures import CaseInsensitiveDict

GEOAPI_KEY = "57e88c5179064b9db5ccd0973355973a"
NSRDBAPI_KEY = "1cJyHKd9AYLbDbCQxt8RJTxIuhIHFI9AIRHbCr76"
newAddress = "f"

print("Predicting Solar Power Using the NSRDB")
print("This program will allow you to determine a timeline for paying off your solar installation. We begin by predicting the amount of solar power that can be generated at \
    the entered location. We then use the expected installation cost and the average electricity cost in your region to let you know when you can expect to start turning a profit!")

def main():
    year = 2020
    attributes1 = 'air_temperature'
    attributes2 = 'alpha'
    attributes3 = 'aod'
    attributes4 = 'asymmetry'
    attributes5 = 'cld_opd_dcomp'
    attributes6 = 'cld_reff_dcomp'
    attributes7 = 'clearsky_dhi'
    attributes8 = 'clearsky_dni'
    attributes9 = 'clearsky_ghi'
    attributes10 = 'cloud_press_acha'
    attributes11 = 'cloud_type'
    attributes12 = 'dew_point'
    attributes13 = 'dhi'
    attributes14 = 'dni'
    attributes15 = 'ghi'
    attributes16 = 'ozone'
    attributes17 = 'relative_humidity'
    attributes18 = 'solar_zenith_angle'
    attributes19 = 'ssa'
    attributes20 = 'surface_albedo'
    attributes21 = 'surface_pressure'
    attributes22 = 'total_precipitable_water'
    attributes23 = 'wind_direction'
    attributes24 = 'wind_speed'
    leap_year = 'false'
    interval = '30'
    utc = 'false'
    your_name = 'None+None'
    reason_for_use = 'beta+testing'
    your_affiliation = 'Springboard'
    your_email = 'edwardjs43@gmail.com'
    mailing_list = 'false'

    x_train = pd.read_csv('../data/X_train_New.csv')
    y_train = pd.read_csv('../data/Y_train_New.csv')
    Electricity_cost = pd.read_csv('../data/StatesElectricity.csv')
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
    url3 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes7)
    url4 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes8)
    url5 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes9)
    url6 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes11)
    url7 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes12)
    url8 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes13)
    url9 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes14)
    url10 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes15)
    url11 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes17)
    url12 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes18)
    url13 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes20)
    url14 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes21)
    url15 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes22)
    url16 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes23)   
    url17 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=NSRDBAPI_KEY, attr=attributes24)   
    
    Locationdf1 = pd.read_csv(url2)
    Locationdf2 = pd.read_csv(url3)
    Locationdf3 = pd.read_csv(url4)
    Locationdf4 = pd.read_csv(url5)
    Locationdf5 = pd.read_csv(url6)
    Locationdf6 = pd.read_csv(url7)
    Locationdf7 = pd.read_csv(url8)
    Locationdf8 = pd.read_csv(url9)
    Locationdf9 = pd.read_csv(url10)
    Locationdf10 = pd.read_csv(url11)
    Locationdf11 = pd.read_csv(url12)
    Locationdf12 = pd.read_csv(url13)
    Locationdf13 = pd.read_csv(url14)
    Locationdf14 = pd.read_csv(url15)
    Locationdf15 = pd.read_csv(url16)
    Locationdf16 = pd.read_csv(url17)

    def Cleandf(df):
        df = df.iloc[:,:6]
        df.rename(columns = df.loc[1], inplace = True)
        df.drop([0,1], inplace = True)
        df.reset_index(drop = True, inplace = True)
        df.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True)
        return df

    Locationdf1 = Cleandf(Locationdf1)
    Locationdf2 = Cleandf(Locationdf2)
    Locationdf3 = Cleandf(Locationdf3)
    Locationdf4 = Cleandf(Locationdf4)
    Locationdf5 = Cleandf(Locationdf5)
    Locationdf6 = Cleandf(Locationdf6)
    Locationdf7 = Cleandf(Locationdf7)
    Locationdf8 = Cleandf(Locationdf8)
    Locationdf9 = Cleandf(Locationdf9)
    Locationdf10 = Cleandf(Locationdf10)
    Locationdf11 = Cleandf(Locationdf11)
    Locationdf12 = Cleandf(Locationdf12)
    Locationdf13 = Cleandf(Locationdf13)
    Locationdf14 = Cleandf(Locationdf14)
    Locationdf15 = Cleandf(Locationdf15)
    Locationdf16 = Cleandf(Locationdf16)
    
    Locationdf = Locationdf1.merge(Locationdf2, left_index = True, right_index = True).merge(Locationdf3, left_index = True, right_index = True).merge(Locationdf4, left_index = True, right_index = True) \
        .merge(Locationdf5, left_index = True, right_index = True).merge(Locationdf6, left_index = True, right_index = True).merge(Locationdf7, left_index = True, right_index = True) \
        .merge(Locationdf8, left_index = True, right_index = True).merge(Locationdf9, left_index = True, right_index = True).merge(Locationdf10, left_index = True, right_index = True) \
        .merge(Locationdf11, left_index = True, right_index = True).merge(Locationdf12, left_index = True, right_index = True).merge(Locationdf13, left_index = True, right_index = True) \
        .merge(Locationdf14, left_index = True, right_index = True).merge(Locationdf15, left_index = True, right_index = True).merge(Locationdf16, left_index = True, right_index = True)

    columnNames = {'Temperature':'air_temperature','Clearsky DHI':'clearsky_dhi','Clearsky DNI':'clearsky_dni',
    'Clearsky GHI':'clearsky_ghi','Cloud Press Acha':'cloud_press_acha','Cloud Type':'cloud_type','Dew Point':'dew_point',
    'DHI':'dhi','DNI':'dni','GHI':'ghi','Relative Humidity':'relative_humidity','Solar Zenith Angle':'solar_zenith_angle','SSA':'ssa',
    'Surface Albedo':'surface_albedo','Pressure':'surface_pressure','Precipitable Water':'total_precipitable_water','Wind Direction':'wind_direction',
    'Wind Speed':'wind_speed'}

    Locationdf.rename(columns = columnNames, inplace = True)

    y_train.drop(columns = ['Unnamed: 0'], inplace = True)
    x_train.drop(columns = ['Unnamed: 0'], inplace = True)

    model = make_pipeline(StandardScaler(), PCA(n_components = 16), RandomForestRegressor(n_estimators = 50, max_depth = 16))

    Electricity_cost.set_index('State', inplace = True)
    Electricity_cost.drop(columns = 'Unnamed: 0', inplace = True)

    for col in Locationdf.columns:
        Locationdf[col] = Locationdf[col].astype(float)

    Locationdf.dropna(inplace = True)

    model.fit(x_train, y_train)

    Locationdf['Power'] = model.predict(Locationdf) 
    Locationdf['Power'] = Locationdf['Power'] * (InstallationSize / 10000)

    StateElectricityCost1 = Electricity_cost.loc[state1]['Electricity Prices']

    Locationdf['Money_saved'] = Locationdf['Power'] * (StateElectricityCost1 / 2)

    dollars_saved1 = Locationdf['Money_saved'].sum() / 100

    print('The amount that you would save per year at your location is $' + str(dollars_saved1))
    TimeTillPayed = InstallationCost / dollars_saved1
    print('By the estimation, you would be aple to pay off your solar installation in ' + str(TimeTillPayed) + ' years!')

    Locationdf['Date'] = pd.date_range(start = "01/01/22 00:00:00", periods = Locationdf.shape[0], freq = '30T')
    Locationdf['Month'] = Locationdf['Date'].agg(func = lambda x: x.month)

    fig, ax = plt.subplots(figsize = (6, 4))
    ax.hist(Locationdf['Power'])
    ax.set(xlabel='Power Generated', ylabel='Count', title = 'Histogram of Power Generated')

    plt.show()

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

    scatterplots(['Power', 'air_temperature', 'dhi', 'dew_point', 'relative_humidity', 'surface_pressure'], figsize = (14, 4))

    plt.show()

Address =  "1600 Pennsylvania Avenue , Washington DC , 20500"

tempAddress = Address.rsplit(" ")
Address = ""
Address += tempAddress[0]
for idx, word in enumerate(tempAddress, start = 1):
    if word == ',':
        Address +=  '%2C'
    else:
        Address += '%20' + word

InstallationCost = 2400

InstallationSize = 0.6

main()