from meteostat import Daily, Stations
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from dateutil.relativedelta import relativedelta

class WeatherData:
  weather_stations = None
  weather_data = None
  wiki_stations = None
  def __init__(self):
    self.output_dir = os.path.join('data', 'raw')
    self.start_date = datetime(2018, 1, 1)

    if WeatherData.wiki_stations is None:
      WeatherData.wiki_stations = self._collect_wiki_stations()
    if WeatherData.weather_stations is None:
      WeatherData.weather_stations = self.collect_weather_stations()
    if WeatherData.weather_data is None:
      WeatherData.weather_data = self.collect_data_by_weather_station()

  def __str__(self):
    return "This is a function that extracts weather data per call sign"

  def _clean_office(self, x):
    x = x.split('[')[0]
    if ' - ' in x:
      x = x.split(' - ')[0]
    if '(' in x and ')' in x:
      left = x.index('(')
      right = x.index(')')
      x = x[left+1:right]
    if "/" in x:
      x = x.split("/")[0]
    if "“" in x:
      x = x.split("“")[-1]
    if '-' in x:
      x = x.split('-')[0]
    return x

  def _collect_wiki_stations(self):
    wiki_url = "https://en.wikipedia.org/wiki/List_of_National_Weather_Service_Weather_Forecast_Offices"
    wiki_path = os.path.join(self.output_dir, 'wiki_stations.csv')
    if os.path.exists(wiki_path):
        return pd.read_csv(wiki_path)
    wiki_stations = pd.concat(pd.read_html(wiki_url)[:-1], axis = 0)
    wiki_stations['Clean office'] = wiki_stations['Forecast office'].apply(self._clean_office)
    wiki_stations['State abbr'] = wiki_stations['Address'].apply(lambda x: x.split(' ')[-2])
    return wiki_stations

  def _convert_station_callsigns(self, data):
    station_frames = []
    for i, row in tqdm(WeatherData.wiki_stations.iterrows()):
      city = row['Clean office']
      region = row['State abbr']
      temp_data = data.loc[(data.name.str.contains(city)) & (data.region.str.contains(region))] 
      temp_data['callsign'] = f"{city}, {region}"
      station_frames.append(temp_data)
    data = pd.concat(station_frames, axis = 0, ignore_index = True)
    return data

  def _interpolate_data(self, data):
    call = data.callsign.unique()
    min_date = data.date.min()
    max_date = data.date.max()
    date_range = pd.date_range(min_date, max_date)
    new_weather= data.set_index(['callsign', 'date'])
    new_index = pd.MultiIndex.from_product([call, date_range])
    new_weather = new_weather.reindex(new_index, axis = 0)
    new = new_weather.reset_index().rename({'level_0': 'callsign', 'level_1': 'date'}, axis = 1)
    interpolated = []
    for u in new.callsign.unique():
      temp = new[new.callsign == u]
      interp = temp.iloc[:, 2:].interpolate().bfill().ffill().fillna(0)
      first = temp[['callsign', 'date']]
      together = pd.concat([first, interp], axis = 1)
      interpolated.append(together)
    new_weather = pd.concat(interpolated, axis = 0)
    new_weather = new_weather.drop(['tsun', 'wpgt'], axis = 1)
    return new_weather

  # collect weather stations
  def collect_weather_stations(self) -> pd.DataFrame:
    output_path = os.path.join(self.output_dir, 'weather_stations.csv')
    if not os.path.exists(output_path):
      stations = Stations().region('US').fetch()
      stations = stations[(stations['daily_start'] <= '2018-01-01') & (stations['daily_end'] >= '2022-01-01')].reset_index()
      names = stations['name']
      regions = stations['region']
      stations = pd.concat([stations.loc[(names.str.contains(row['Clean office'])) & (regions.str.contains(row['State abbr'])), :] for i, row in WeatherData.wiki_stations.iterrows()],
                                ignore_index = True)
      stations = self._convert_station_callsigns(stations)
      stations.to_csv(output_path, index = False)
    else:
      stations = pd.read_csv(output_path)
    return stations

  def _collect_data_by_weather_station_helper(self, date):
    list_of_frames = []
    for i, row in tqdm(WeatherData.weather_stations.iterrows()):
      temp = Daily(str(row['id']), date, datetime.now())
      temp.threads = 4
      temp = temp.fetch().reset_index().rename({'time': 'date'}, axis = 1)
      for col in ['name', 'region']:
        temp[col] = row[col]
      list_of_frames.append(temp)
    total_frames = pd.concat(list_of_frames, axis = 0, ignore_index = True)
    total_frames['date'] = total_frames['date'].astype(str)
    total_frames = self._convert_station_callsigns(total_frames).groupby(['callsign', 'date']).mean().reset_index()
    total_frames['date'] = pd.to_datetime(total_frames['date'])
    total_frames = total_frames.set_index(['callsign', 'date']).reset_index()

    # fix missing data and interpolate by callsign
    new_weather = self._interpolate_data(total_frames)
    return new_weather

  def collect_data_by_weather_station(self):
    output_path = os.path.join(self.output_dir, 'weather_data_by_station.csv')
    if not os.path.exists(output_path):
      total_frames = self._collect_data_by_weather_station_helper(self.start_date)
      total_frames.to_csv(output_path, index = False)
    else:
      total_frames = pd.read_csv(output_path, parse_dates = ['date'])
      last_date = total_frames['date'].max() - relativedelta(days = 1)
      if last_date < datetime.now().date()-relativedelta(days = 1):
        print("Updating weather by stations dataset...")
        new_data = self._collect_data_by_weather_station_helper(date = last_date)
        new_data = self._interpolate_data(new_data)
        total_frames = pd.concat([total_frames, new_data], axis = 0, ignore_index = True)
        total_frames['date'] = pd.to_datetime(total_frames['date'])
        total_frames.to_csv(output_path, index = False)

    return total_frames
  
  def _calc_end_range(self, date, num_prev: int):
    return date + pd.to_timedelta(f"{num_prev} days") + pd.to_timedelta("29 days")

  def ml_data(self, start_date: str, num_prev: int = 365):
    start_date = pd.to_datetime(start_date)

    eventual = start_date + relativedelta(days=num_prev + 30)
    if eventual > datetime.now():
      raise ValueError(f"Given num_prev and start_date ({eventual}) cannot exceed current date: {datetime.now().date()}")
    first_filtered = WeatherData.weather_data.copy()
    first_filtered = first_filtered[first_filtered['date'] >= start_date]
    data = first_filtered.melt(id_vars = ['callsign', 'date'])
    data['date'] = pd.to_datetime(data['date'])
    callsign_measure = list(set(zip(data['callsign'], data['variable'])))

    thirtydays = pd.to_timedelta("30 days")
    one_day = pd.to_timedelta("1 day")

    new_values, new_labels = [], []
    for c, m in tqdm(callsign_measure):
      temp = data.loc[(data['callsign'] == c) & (data['variable'] == m), ['date', 'value']].set_index('date')
      min_date = temp.index.min().date()
      max_date = temp.index.max().date()
      collection = {'X': [], 'y': []}
      end_range = self._calc_end_range(min_date, num_prev = num_prev)
      while max_date != end_range:
        row = temp.loc[min_date:end_range]
        splits = end_range - thirtydays
        values = np.array(row[min_date:splits])
        labels = np.array(row[splits+one_day:])
        collection['X'].append(values.reshape(1, -1))
        collection['y'].append(labels.reshape(1, -1))

        min_date += one_day
        end_range = self._calc_end_range(min_date, num_prev = num_prev)
      collection['X'] = pd.DataFrame(np.stack(collection['X']).reshape(-1, num_prev))
      collection['y'] = pd.DataFrame(np.stack(collection['y']).reshape(-1, 30))
      # add the callsigns and measures
      for dat in ['X', 'y']:
        for string_val, val in zip(['callsign', 'measure'], [c, m]):
          collection[dat][string_val] = val
      new_values.append(collection['X'])
      new_labels.append(collection['y'])

    values = pd.concat(new_values, ignore_index = True)
    labels = pd.concat(new_labels, ignore_index = True)

    return values, labels