# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:43:28 2022

@author: Collin
"""

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
#import dash_core_components as dcc
#import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
from utils import WeatherData
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from arima import WeatherARIMA, ARIMAPrediction, join_station_measure
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# get the stations
wd = WeatherData()

measure_conversions = {"tavg": 'Average Temperature (°C)',
                       'tmin': 'Minimum Temperature (°C)',
                       'tmax': 'Maximum Temperature (°C)',
                       'prcp': 'Precipitation (mm)',
                       'snow': 'Snowfall (mm)',
                       'wdir': 'Wind Direction (Degrees)',
                       'wspd': 'Wind Speed (km/hr)',
                       'pres': 'Air Pressure (hPa)'}

def reframe_stations(x):
    y = x.split(', ')
    new = f"{y[1]} - {y[0]}"
    return new

def weather_stations(weather_instance = wd):
    stations = weather_instance.weather_stations
    new_stations = pd.DataFrame()
    for col in ['value', 'label']:
        new_stations[col] = stations['callsign']
    new_stations['label'] = new_stations['label'].apply(lambda x: reframe_stations(x))
    new_data = new_stations.drop_duplicates().sort_values('label')
    id_callsign = new_data.to_dict('records')
    return id_callsign

def enviromental_measures(weather_instance = wd):
    values = weather_instance.weather_data
    measures = list(values.columns[2:])
    new_measures = pd.DataFrame()
    for col in ['value', 'label']:
        new_measures[col] = measures
    new_measures['label'] = new_measures['label'].apply(lambda x: measure_conversions[x])
    id_measure = new_measures.to_dict('records')
    return id_measure

# get weather stations
stations = weather_stations()
measures = enviromental_measures()
        
# state stylesheets
external_stylesheets = [dbc.themes.SIMPLEX]
app = Dash(__name__, external_stylesheets = external_stylesheets)

current_date = pd.to_datetime(datetime.now().date())

# app layout
# header
row_header = html.Div(
    dbc.Row([
        dbc.Col([
            html.Div(
                [
                    html.H1('Environmental Measure Forecasting Application')
                ]
            )
        ])
    ])
)

row_info = html.Div(
    dbc.Row([
        dbc.Col([
            html.P("""This application allows for forecasting different environmental
                  measures using different models, ARIMA and LSTM. Choose which model works 
                  best for you!
                  
                  This is a final project for Times Series Analysis at Indiana University.
                  """
            )
        ])
    ])
)

# dropdown menus with labels
row_dropdowns = html.Div(className = 'two columns', children = [
     dbc.Row([
        # model container
        dbc.Col([
            dbc.Label('Models:'),
            dcc.Dropdown(
                id='dropdownModels',
                options=[
                    {'label': 'LSTM', 'value': 'lstm'},
                    {'label': 'ARIMA', 'value': 'arima'},
                ],
                clearable=False,
                searchable=False,
                multi = True,
                style = {"align-items": "center",
                         'justify-content': 'center'}
            )   
        ], width = {'order': 'first'}, align = 'center'
        ),
            
        # station container
        dbc.Col([
            dbc.Label("Station:"),
            dcc.Dropdown(
                id='dropdownStations',
                options=stations,
                value=stations[0]['value'],
                clearable=False,
                searchable=True,
                style = {'text-align': 'center',
                         "align-items": "center",
                         'justify-content': 'center'}
            )
        ], align = 'center'
        ),
        dbc.Col([
            dbc.Label("Measure:"),
            dcc.Dropdown(
                id='dropdownMeasure',
                options=measures,
                value='tavg',
                clearable=False,
                searchable=True,
                style = {'text-align': 'center',
                         "align-items": "center",
                         'justify-content': 'center'}
            )
        ], align = 'center'
        ),
        dbc.Col([
            dbc.Label("Time Before:"),
            dcc.Dropdown(
                id='dropdownTime',
                options = [{"value": "week_2", "label": "2 Weeks"},
                           {"value": "month_1", "label": "1 Month"},
                           {"value": "month_3", 'label': "3 Months"},
                           {"value": "month_6", 'label': "6 Months"},
                           {"value": "year_1", 'label': "1 Year"},
                           {"value": "all_1", 'label': 'All'}],
                value = "week_2",
                clearable = False,
                searchable = True,
                style = {'text-align': 'center',
                         "align-items": "center",
                         'justify-content': 'center'}
            )
        ], align = 'center'
        )
    ])
])            

row_figure = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'figureForecast', style = {"width": "99%"}),
        ]),
        dbc.Col([
            dcc.Graph(id = 'figureSeasonal', style = {"width": "99%"})
        ])
    ])
])

row_slider = html.Div([
    dbc.Row([
        dbc.Label("Days to Forecast:", html_for = 'slider'),
        dcc.Slider(id = 'sliderForecast', 
                   min = 1,
                   max = 30,
                   step = 1,
                   value = 7,
                   marks = {i: str(i) for i in range(1, 31)})
    ])],
    className="mb-3"
)

row_test = html.Div(id = 'output-test')

row_footer = html.Div([
    dbc.Row([
        dbc.Col([
            html.P(
                "Web Application: Collin J. Kovacs"
            ),
            html.P(
                "ARIMA Model: Matthew Yeseta"
            ),
            html.P(
                "LSTM Model: Pankaj Singh/Collin J. Kovacs"
            )
        ]),
        dbc.Col([
            html.Img(src = "images/indiana.png",
                     width = '128px',
                     height = '128px')     
        ])
    ])
], style = {"background-color": "gray", 
            "width": "100%"})

app.layout = dbc.Container(children = [
    row_header,
    row_info,
    html.Br(),
    row_dropdowns,
    row_slider,
    html.Hr(),
    row_figure,
    row_test,
    html.Hr(),
    row_footer
])

def create_seasonal_plots(data, measure_name):
    # seasonal plots
    seasonal_plot = seasonal_decompose(data[['date', measure_name]].set_index('date'))
    season_data = pd.DataFrame({'trend': seasonal_plot.trend, 
                                'seasonal': seasonal_plot.seasonal, 
                                'resid': seasonal_plot.resid})
    season_data = season_data.reset_index()
    season_data = season_data.melt(id_vars=['date']).rename({'variable': "Variables"}, axis = 1)
    fig = px.line(data_frame = season_data, x = "date", y = "value", facet_row="Variables", hover_name="Variables", hover_data=["value"],
    labels={"value": "Value", 'date': "Date [1D]"}, title="Seasonal Decomposition of Additive Base Model")
    fig.update_yaxes(title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
                
@app.callback(
    Output('figureForecast', 'figure'),
    Output('figureSeasonal', 'figure'),
    Input('dropdownModels', 'value'),
    Input('dropdownStations', 'value'),
    Input('dropdownMeasure', 'value'),
    Input('dropdownTime', 'value'),
    Input('sliderForecast', 'value'),
    )
def update_graph(model_name, station_name, measure_name, time_name, slider_value):
    subset_data = wd.weather_data.loc[wd.weather_data['callsign'] == station_name, ['date', measure_name]]
    now = datetime.now().date()
    time_splits = time_name.split('_')
    if time_splits[0] == 'all':
        prev_date = subset_data['date'].min()
    else:
        if time_splits[0] == 'week':
            prev = relativedelta(weeks = int(time_splits[1]))
        elif time_splits[0] == 'month':
            prev = relativedelta(months = int(time_splits[1]))
        else:
            prev = relativedelta(years = int(time_splits[1]))
        prev_date = pd.to_datetime(now - prev)
    
    subset_data = subset_data[subset_data['date'] >= prev_date]
    subset_data['type'] = 'History'
    
    # fix if the model is in list but only one model name
    if isinstance(model_name, list):
        if len(model_name) == 1:
            model_name = model_name[0]
    
    # if the user selectes arima
    join_name = join_station_measure(station_name, measure_name)
    if 'arima' == model_name:
        dates = pd.date_range(now, pd.to_datetime(now + relativedelta(days = 30)))
        last_row = subset_data.tail(1)
        arimaprediction = ARIMAPrediction(station_name, measure_name)
        predictions = arimaprediction.predictions.get(join_name)
        arima_data = pd.DataFrame({'date': dates, measure_name: predictions})
        arima_data['type'] = 'Prediction'
        arima_data = arima_data.iloc[:slider_value]
        new_data = pd.concat([subset_data, arima_data], axis = 0)
        
        # actually data
        fig = px.line(new_data, x = 'date', y = measure_name, color = 'type',
                      labels = {measure_name: measure_conversions[measure_name],
                                'date': 'Date [1D]'},
                      title = f"{measure_conversions[measure_name]} by Date for {station_name}", 
                      markers = True)
    
    # if no model selected
    else:
        # actually data
        fig = px.line(subset_data, x = 'date', y = measure_name,
                      labels = {measure_name: measure_conversions[measure_name],
                                'date': 'Date [1D]'},
                      title = f"{measure_conversions[measure_name]} by Date for {station_name}", markers = True)
    
    
    fig1 = create_seasonal_plots(subset_data, measure_name)
    return fig, fig1

@app.callback(
    Output('output-test', 'children'),
    Input('dropdownModels', 'value'),
    Input('dropdownStations', 'value'),
    Input('dropdownMeasure', 'value'),
    Input('dropdownTime', 'value'),
    Input('sliderForecast', 'value')
    )
def ouput_test(model_name, station_name, measure_name, time_name, slider_value):
    return f"{model_name}, {station_name}, {measure_name}, {time_name}, {slider_value}"


if __name__ == '__main__':
    app.run_server(debug=True)
