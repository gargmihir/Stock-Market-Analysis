
# coding: utf-8

# # Stock Market Analysis
# ## Mihir Garg 

# In[1]:

#Importing Libraries
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# For reading stock data 
from pandas_datareader import data

# For time stamps
from datetime import datetime

# Resize the size of plots
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 14
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


# ### Basic Stock Market Analysis

# In[2]:

# The stocks for analysis
stocks=['AAPL','GOOG','MSFT','AMZN']


# In[3]:

# End and Start times for data grab
end_date=datetime.now()
start_date=datetime(end_date.year-1,end_date.month,end_date.day)


# In[4]:

# For grabing yahoo finance data 
for stock in stocks:
    globals()[stock]=data.get_data_yahoo(stock,start_date,end_date)


# In[5]:

# View Apple stock
AAPL.head()


# In[6]:

# View Google stock
GOOG.head()


# In[7]:

# Summary Stats
AAPL.describe()


# In[8]:

# General info
AAPL.info()


# In[9]:

# Historical view for closing price
AAPL['Adj Close'].plot(legend=True,figsize=(15,6),color='red')


# In[10]:

# Total volume of stock being traded each day over the past year
AAPL['Volume'].plot(legend=True,figsize=(15,6),color='red')


# In[11]:

# Calculate Moving Average
moving_average=[10,20,50] # 10 day, 20 day, 50 day moving average

for ma in moving_average:
    column_name = 'MA for %s days' %(str(ma))
    AAPL[column_name] = pd.rolling_mean(AAPL['Adj Close'],ma)


# In[12]:

# Plot of all moving average
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(15,6),colormap='jet')


# ### Daily Return Analysis

# In[13]:

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()


# In[14]:

AAPL.head()


# In[15]:

# Plot daily return
AAPL['Daily Return'].plot(figsize=(15,6),marker='s',color='red',legend=True)


# In[16]:

# Histogram of average daily return
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='green')


# In[17]:

# Analyze all the stocks in list
closing = data.get_data_yahoo(stocks,start_date,end_date)['Adj Close']


# In[18]:

closing.head()


# In[19]:

# Daily return of all stocks
stocks_return=closing.pct_change()
stocks_return.head()


# In[20]:

# Compare the daily percentage return of two stocks to check how correlated
# First GOOG with GOOGL to get perfect linear relationship

sns.jointplot('GOOG','GOOG',data=stocks_return,color='purple',size=10)


# In[21]:

# Plot of GOOG with MSFT
sns.jointplot('GOOG','MSFT',data=stocks_return,color='seagreen',size=10)


# In[22]:

# Plot of correlation of all stocks with each other
sns.pairplot(stocks_return.dropna(),size=4)


# In[23]:

# More in-dept visulization
retrun_plot=sns.PairGrid(stocks_return.dropna(),size=4)

# To modify upperhalf triangle of pair plot
retrun_plot.map_upper(plt.scatter,color='green')

# To modify lowerhalf triangle of pair plot
retrun_plot.map_lower(sns.kdeplot)

# To modify diagonal of pair plot
retrun_plot.map_diag(plt.hist,bins=50)



# In[24]:

# Correlation of closing price

retrun_plot=sns.PairGrid(closing.dropna(),size=5)

# To modify upperhalf triangle of pair plot
retrun_plot.map_upper(plt.scatter,color='purple')

# To modify lowerhalf triangle of pair plot
retrun_plot.map_lower(sns.kdeplot)

# To modify diagonal of pair plot
retrun_plot.map_diag(plt.hist,bins=30)


# In[25]:

# Actual numerical value for correlation of all the stocks return value

sns.heatmap(stocks_return.dropna().corr(),annot=True)


# In[26]:

# Actual numerical value for correlation of all the stocks closing price

sns.heatmap(closing.dropna().corr(),annot=True,cmap='summer')


# ### Risk Analysis

# In[27]:

rets = stocks_return.dropna()


# In[28]:

area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

# Label the scatter plot
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label,xy = (x, y), xytext = (50, 50),textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# ### Value at Risk

# In[29]:

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[30]:

# The 0.05 empirical quantile of daily returns
rets['AAPL'].quantile(0.05)


# The 0.05 empirical quantile of daily returns is at -0.017. That means that with 95% confidence, our worst daily loss will not exceed 1.5%. If we have a Rs. 1,00,000 investment, our daily loss will not excedd 0.015 * 1,00,000 = Rs. 1500 in any case.
# 

# In[31]:

# Monte Carlo method to analysis risk of Apple stock

days=365
dt=1/days
mu=rets.mean()['AAPL']
sigma=rets.std()['AAPL']

# Create a function that takes in the starting price, number of days and uses sigma and mu 

def stock_monte_carlo(start_price,days,mu,sigma):
        
    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for x in range(1,days):
        
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


# In[32]:

AAPL.head()


# In[33]:

start_price=116.519997

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Apple')


# In[34]:

# Histogram of the end results for a much larger run
runs = 10000

# Empty matrix to hold the end price data
simulations = np.zeros(runs)

for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];


# In[35]:

# Histogram
# Define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Plot the distribution of the end prices
plt.hist(simulations)

# plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)

# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Apple Stock after %s days" % days, weight='bold');


# In[ ]:



