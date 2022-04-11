from arima import ARIMAPrediction
ar = ARIMAPrediction('Pittsburgh, PA', 'tavg')
print(ar.predictions)