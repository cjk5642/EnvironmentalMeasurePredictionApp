# install pyPRISM 
from meteostat import Daily, Point
import pandas as pd
import os
from datetime import datetime
from geopy.geocoders import Nominatim
import time
from tqdm.notebook import tqdm
import numpy as np

collin_path = "/content/drive/MyDrive/Classes/Spring 2022/DSCI 590: Time Series Analysis/Final Project/data/raw/indiana/in_prism_climate.gdb"
matt_path = "..."
pankaj_path = "..."

class WeatherData:
  def __init__(self, user_agent = 'cokovacs'):
    self.geolocator = Nominatim(user_agent = user_agent, timeout = None)
    self.output_dir = "/content/drive/MyDrive/Classes/Spring 2022/DSCI 590: Time Series Analysis/Final Project/data/raw"

    self.weather_stations = self.collect_weather_stations()
    self.weather_data = self.collect_data_by_weather_station()

  def __str__(self):
    return "This is a function that extracts weather data per call sign"

  # collect latitude and longitude
  def _collect_latitude_longitude(self, address: str):
    time.sleep(1)
    location = self.geolocator.geocode(address)
    try:
      s = f"{location.latitude},{location.longitude}"
    except:
      s = f"{-1111},{-1111}"
    return s

  # collect weather stations
  def collect_weather_stations(self) -> pd.DataFrame:
    output_path = os.path.join(self.output_dir, 'weather_stations.csv')
    if not os.path.exists(output_path):
      url = "https://en.wikipedia.org/wiki/List_of_National_Weather_Service_Weather_Forecast_Offices"
      tables = pd.read_html(url)[:-1]
      all_tables = pd.concat(tables, axis = 0, ignore_index = True)
      all_tables["lat_long"] = [self._collect_latitude_longitude(row['Address']) for i, row in tqdm(all_tables.iterrows())]
      all_tables[['latitude', 'longitude']] = all_tables['lat_long'].str.split(',', expand = True)
      for c in ['latitude', 'longitude']:
        all_tables[c] = all_tables[c].astype(float)
      all_tables = all_tables.replace(-1111, np.NaN)
      all_tables = all_tables.drop('lat_long', axis = 1).dropna().reset_index(drop = True)
      all_tables.to_csv(output_path, index = False)
    else:
      all_tables = pd.read_csv(output_path)
    return all_tables

  def collect_data_by_weather_station(self):
    start = datetime(2018, 1, 1)
    end = datetime.now()

    output_path = os.path.join(self.output_dir, 'weather_data_by_station.csv')
    if not os.path.exists(output_path):
      list_of_frames = []
      for i, row in tqdm(self.weather_stations.iterrows()):
        lat = row.latitude
        lon = row.longitude
        point = Point(lat, lon)
        temp = Daily(point, start, end).fetch().reset_index().rename({'time': 'date'}, axis = 1)
        for col in ['Forecast office', 'State', 'Office call sign', 'Address', 'latitude', 'longitude']:
          temp[col] = row[col]
        list_of_frames.append(temp)
      total_frames = pd.concat(list_of_frames, axis = 0, ignore_index = True)
      total_frames['date'] = total_frames['date'].astype(str)
      total_frames = total_frames.set_index(['Office call sign', 'date']).fillna(0)
      total_frames.to_csv(output_path)
    else:
      total_frames = pd.read_csv(output_path, index_col=['Office call sign', 'date'])

    return total_frames
  
  @property
  def ml_data(self):
    sub_data = self.weather_data.iloc[:, :10].reset_index().ffill()
    call_sign_dummies = pd.get_dummies(sub_data['Office call sign'])
    dummy_data = pd.concat([sub_data.drop('Office call sign', axis = 1), call_sign_dummies], axis = 1)
    return dummy_data