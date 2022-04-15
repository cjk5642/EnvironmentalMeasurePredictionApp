import pandas as pd
from utils import WeatherData
weather = WeatherData(update = False)
values, labels = weather.ml_data("01-01-2022", num_prev=30)
print(values)


