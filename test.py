from utils import WeatherData

data = WeatherData(update=False).ml_data("01/01/2022", 30, normalize_by_values=True)
print(data)

