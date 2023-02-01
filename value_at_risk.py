# First of all we have to import the libraries that we are going to use

from scipy.stats import norm # Because we are going to work with distributions
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

yf.pdr_override()

# Then we have to download the data

tickers = ['AAPL', 'TSLA']
weights = np.array([0.5,0.5])
startdate = '2012-01-01'
enddate = '2023-02-01'
data = pd.DataFrame()

for x in tickers:
    data[x] = pdr.get_data_yahoo(x, start=startdate, end=enddate)['Adj Close']

# Values

returns = data.pct_change() #Simple returns
cov_matrix = returns.cov()
returns_mean = returns.mean()
portfolio_mean = returns_mean.dot(weights)
portfolio_std = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

# Variables

investment = float(100000)

# Model

mean_investment = (1+portfolio_mean)*investment
std_investment = investment * portfolio_std
conf_level = 0.05
cut = norm.ppf(conf_level, mean_investment, std_investment)
var_id = investment - cut
num_days = int(100)

var_array = []

print("\nThe maximum loss of your portfolio with a $" + str(investment) + " investment" + " \nand with a " + str((1-conf_level)*100) + "% confidence level for the next " + str(num_days) + " days is:")

for i in range(1, num_days):
    var_array.append(np.round(var_id*np.sqrt(i), 2))
    print(str(i) + ( " days. VaR(" + str((1-conf_level)*100) + "%) = " + str((np.round(var_id*np.sqrt(i), 2)))))

# Finally, we can graph these results

plt.xlabel("Days")
plt.ylabel("The Maximum loss of your portfolio")
plt.title("The Maximum loss of your portfolio for the period")
plt.plot(var_array, "b")
plt.show()