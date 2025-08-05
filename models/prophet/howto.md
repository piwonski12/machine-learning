
# How to import 

if you ever want to check out these useless models


## Import and usage

Python import and execution

```bash
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


#model = joblib.load('[PKL FILE HERE!!!]') as in the example below
model = joblib.load('prophet_sales_pretrained_model.pkl')

#(periods=100) is the period of time in the future that you want to predict
future = model.make_future_dataframe(periods=100)
forecast = model.predict(future)

#just plot prediction visualization
model.plot(forecast)
plt.show()
```
Export prediction as csv file

```bash
prediction_file_name = 'forecast.csv'
forecast.to_csv(prediction_file_name,  index=False)
