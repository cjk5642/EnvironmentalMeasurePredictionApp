# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:05:08 2022

@author: Collin
"""
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
from app import lstm_model, wd
from dateutil.relativedelta import relativedelta

def join_station_measure(station, measure):
    station = '-'.join(station.split(', ')).lower()
    return '_'.join([station, measure])

class WeatherLSTM:
    model = None
    def __init__(self, callsign: str, measure: int, model = lstm_model, weatherdata = wd):
        self.callsign = callsign
        self.measure = measure
        self.weatherdata = wd
        self.model = model
        self.weather_data = self._extract_weather_data()
        self.ohe_callsign = self._extract_ohe_callsign()
        self.ohe_measures = self._extract_ohe_measure()
    
    def _extract_weather_data(self):
        data = self.weatherdata.weather_data.copy()
        data = data.loc[data['callsign'] == self.callsign, self.measure].reset_index(drop = True).iloc[-366:].values.reshape(1, -1).tolist()
        self.mean = np.mean(data)
        self.std = np.std(data)
        data = (np.array(data) - self.mean) / self.std
        return data

    def _extract_ohe_callsign(self):
        return pd.read_csv(r"./data/raw/ohe_callsigns.csv", usecols = [self.callsign]).values.reshape(1, -1)
    
    def _extract_ohe_measure(self):
        return pd.read_csv(r"./data/raw/ohe_measures.csv", usecols = [self.measure]).values.reshape(1, -1)
        
    def predict(self):
        joined = tf.concat([self.weather_data, self.ohe_callsign, self.ohe_measures], axis = 1)
        reshaped_joined = tf.reshape(joined, shape = (1, -1))
        output = self.model.predict(reshaped_joined)
        output = np.array(output*self.std + self.mean).tolist()[0]
        return output
    
class LSTMPrediction:
    predictions = {}
    def __init__(self, station_name, measure_name):
        joined = join_station_measure(station_name, measure_name)
        if LSTMPrediction.predictions.get(joined) is None:
            lstm = WeatherLSTM(station_name, measure_name)
            LSTMPrediction.predictions[joined] = lstm.predict()

class WeatherARIMA:
    models = None
    def __init__(self, callsign: str, measure: int, weatherdata = wd):
        self.callsign = callsign
        self.measure = measure
        self.model_path = "./data/processed/best_arima_models.csv"
        self.weatherdata = wd
        if WeatherARIMA.models is None:
            WeatherARIMA.models = pd.read_csv(self.model_path)
        
        self.weather_data = self._extract_weather_data()
        self.order = self._extract_order()
    
    def _extract_weather_data(self):
        data = self.weatherdata.weather_data.copy()
        data = data.loc[data['callsign'] == self.callsign, ['date', self.measure]].reset_index(drop = True).set_index('date')
        return data
        
    def _extract_order(self) -> tuple:
        data = WeatherARIMA.models.loc[(WeatherARIMA.models['callsign'] == self.callsign) & (WeatherARIMA.models['measure'] == self.measure), ['p', 'd', 'q']]
        order_values = data.values[0]
        order = list(order_values)
        return order
    
    def predict(self):
        date_split = pd.to_datetime(self.weather_data.index.max()) - relativedelta(days = 29)
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
        return predictions
 
class ARIMAPrediction:
    predictions = {}
    def __init__(self, station_name, measure_name):
        joined = join_station_measure(station_name, measure_name)
        if ARIMAPrediction.predictions.get(joined) is None:
            arima = WeatherARIMA(station_name, measure_name)
            ARIMAPrediction.predictions[joined] = arima.predict()