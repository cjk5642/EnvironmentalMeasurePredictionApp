# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:05:08 2022

@author: Collin
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils import WeatherData
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, cpu_count

weatherdata = WeatherData()

class WeatherARIMA:
    models = None
    def __init__(self, callsign: str, measure: int, weatherdata = weatherdata):
        self.callsign = callsign
        self.measure = measure
        self.model_path = "./data/processed/best_arima_models.csv"
        self.weatherdata = weatherdata
        if WeatherARIMA.models is None:
            WeatherARIMA.models = pd.read_csv(self.model_path)
        
        self.weather_data = self._extract_weather_data()
        self.order = self._extract_order()
    
    def _extract_weather_data(self):
        data = self.weatherdata.weather_data
        data = data.loc[(data['callsign'] == self.callsign), ['date', self.measure]].set_index('date').asfreq("D")
        return data
        
    def _extract_order(self) -> tuple:
        data = WeatherARIMA.models.loc[(WeatherARIMA.models['callsign'] == self.callsign) & (WeatherARIMA.models['measure'] == self.measure), ['p', 'd', 'q']]
        order_values = data.values[0]
        order = tuple(list(order_values))
        return order
    
    def _fit_model(self, data, order):
        pass
    
    def predict(self):
        date_split = pd.to_datetime(datetime.now().date()) - relativedelta(days = 30)
        history, test = self.weather_data[:date_split], self.weather_data[date_split:]
        predictions = []
        history = list(history[self.measure])
        test = list(test[self.measure])
        
        for i in range(len(test)):
            model = ARIMA(history, order = self.order).fit()
            preds = model.forecast(n=30)
            yhat = preds[0]
            predictions.append(yhat)
            obs = test[i]
            history.append(obs)
        predictions = pd.DataFrame(predictions)
        predictions.columns = ['Prediction']
        return predictions
    
def join_station_measure(station, measure):
    station = '-'.join(station.split(', ')).lower()
    return '_'.join([station, measure])
    
class ARIMAPrediction:
    predictions = {}
    def __init__(self, station_name, measure_name):
        joined = join_station_measure(station_name, measure_name)
        if ARIMAPrediction.predictions.get(joined) is None:
            arima = WeatherARIMA(station_name, measure_name)
            ARIMAPrediction.predictions[joined] = arima.predict()