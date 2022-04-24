from models import WeatherLSTM
import numpy as np

wl = WeatherLSTM('Pittsburgh, PA', 'tavg')
print(wl.predict())
