import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

countries = ["DE", "CH", "BE", "NL", "AT", "FR"]

def saturation_model(x, A, k, x0, C):
  return (A / (1 + np.exp(-k*(x-x0))))-C

def profit_model(x, A, k, x0):
  return A*((np.exp(-k*(x-x0))*k)/(1 + np.exp(-k*(x-x0))**2))

def wind_solar_model(x,a,b):
   return a*x+b

dataframe = read_csv("countries_arbitrage.csv", header=None)
day_ahead = dataframe.values

dataframe = read_csv("countries_objective.csv", header=None)
total = dataframe.values

fcr = np.subtract(total, day_ahead)

dataframe = read_csv("countries_vre_capacity_yearly.csv", header=None)
wind_solar_raw = dataframe.values
wind_solar = np.zeros((len(day_ahead), len(day_ahead[0])))
start_offset = 6
for country in range(len(day_ahead)):
  for i in range(len(day_ahead[0])):
    year = math.floor((i+start_offset)/12)
    wind_solar[country,i] = wind_solar_raw[country,year]

for i in range(0,len(countries)):
  # x = np.asfarray(storage[0,:])
  y = np.asfarray(fcr[i,:])
  x = np.cumsum(np.ones(len(y)))

  popt, _ = curve_fit(profit_model, x, y, p0=[17000,0.1,20])
  A, k, x0 = popt
  print('A = %.5f k= %.5f x0=%.5f' % (A, k, x0))
  
  x_line = np.arange(min(x), max(x), 1)
  y_line = profit_model(x_line, A, k, x0)

  y = np.cumsum(fcr[i,:])
  x = np.cumsum(np.ones(len(y)))
  plt.scatter(np.cumsum(np.ones(len(y))), y, label=countries[i]+" data")
  y_line = np.cumsum(profit_model(x, A, k, x0))
  plt.plot(np.cumsum(np.ones(len(y))), y_line, '--', label=countries[i] + " saturation model")
  print("FCR mean deviation (percentage): ", np.mean(np.abs((y_line-y)/y)))
  print("FCR mean deviation (absolute): ", np.mean(np.abs(((y_line-y)))))
  # plt.show()
plt.ylabel("FCR Cumulative Profit (EUR)")
plt.grid()
plt.xlabel("Experiment Month")
plt.title("Cumulative FCR Profit with Intraday Trading")
plt.legend()
plt.show()

for i in range(0,len(countries)):

  x = np.asfarray(wind_solar[i,:])
  y = np.asfarray(day_ahead[i,:])

  popt, _ = curve_fit(wind_solar_model, x, y)
  a,b = popt
  print('a = %.5f b= %.5f' % (a,b))

  x=wind_solar[i,:]
  x_line = np.arange(min(x), max(x), 1)
 
  x = np.asfarray(wind_solar[i,:])
  y = day_ahead[i,:]
  plt.scatter(x, y, label=countries[i]+" data")
  y_line = wind_solar_model(x_line,a,b)
  plt.plot(x_line, y_line, '--', label=countries[i])

  plt.title("Cumulative Arbitrage Profit with Intraday Trading")
  plt.grid()
  plt.legend()
  plt.show()

for i in range(len(countries)):
  y = np.asfarray(fcr[i,:])
  x = np.cumsum(np.ones(len(y)))
  popt, _ = curve_fit(profit_model, x, y, p0=[17000,0.1,20])
  A, k, x0 = popt
  # x = np.asfarray(storage[0,:])
  y = np.cumsum(fcr[i,:])
  x = np.cumsum(np.ones(len(y)))
  fcr_model = np.cumsum(profit_model(x, A, k, x0))

  x = np.asfarray(wind_solar[i,:])
  y = np.asfarray(day_ahead[i,:])
  popt, _ = curve_fit(wind_solar_model, x, y)
  a,b = popt
  day_ahead_model = np.cumsum(wind_solar_model(x,a,b))

  y = np.cumsum(total[i,:])
  plt.scatter(np.cumsum(np.ones(len(y))), y, label=countries[i]+" data")
  plt.plot(np.cumsum(np.ones(len(y))), fcr_model+day_ahead_model, label=countries[i]+" model")
  print(countries[i])
  print("mean deviation (percentage): ", np.mean(np.abs((fcr_model+day_ahead_model-y)/y)))
  print("mean deviation (absolute): ", np.mean(np.abs((fcr_model+day_ahead_model-y))))

plt.title("Cumulative Profit with Intraday Trading")
plt.ylabel("Cumulative Profit (EUR)")
plt.xlabel("Experiment Month")
plt.legend()
plt.grid()
plt.show()