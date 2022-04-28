from utils import WeatherData
import pandas as pd

data = WeatherData(update=True).ml_data("01/01/2022", 30, normalize_by_values=True, normalize_labels=True)
values, labels = data
print(values)
print(labels)