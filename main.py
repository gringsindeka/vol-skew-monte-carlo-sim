
from matplotlib import dates
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.stats import norm


# 1.0-Pulling option chain for a specific ticker

#Setting the ticker
symbol = "MSFT"

def select_exp(symbol):
    tk = yf.Ticker(symbol)
    dates = tk.options
    print(dates)
    return dates


#Default
    #Select expiration date
    #exp_index= 7
    #min_iv = 0.01
    #min_volume = 0

# 1.1-Defining the function that will fetch the option chain from yf
def pull_option_chain(symbol,exp_index,min_iv, min_volume):
    tk = yf.Ticker(symbol)
    dates = tk.options
    target_chain = tk.option_chain(dates[exp_index])
    # Getting the current spot
    spot_price = target_chain.underlying.get('regularMarketPrice')
    calls_df = target_chain.calls
    puts_df = target_chain.puts
    # Filtering 
    calls_df = calls_df[calls_df['volume'] > min_volume]
    puts_df = puts_df[puts_df['volume'] > min_volume]
    calls_df = calls_df[calls_df['impliedVolatility'] > min_iv]
    puts_df = puts_df[puts_df['impliedVolatility'] > min_iv]
    # Slicing columns of interest
    calls = calls_df[['strike', 'lastPrice', 'impliedVolatility']]
    puts = puts_df[['strike', 'lastPrice', 'impliedVolatility']]
    #Adding the spot to Call and Put df 
    calls['spot_price'] = spot_price
    puts['spot_price'] = spot_price
    #Adding Moneyness
    calls['moneyness'] = calls['strike']/calls['spot_price']
    calls['bs_logmoneyness'] = np.log(1/calls['moneyness'])
    puts['moneyness'] = puts['strike']/puts['spot_price']
    puts['bs_logmoneyness'] = np.log(1/puts['moneyness'])
    
    #Plotting (IV, moneyness) for Calls and puts
    plt.plot(calls['moneyness'],calls['impliedVolatility']*100, label ='Calls')
    plt.plot(puts['moneyness'],puts['impliedVolatility']*100, label = 'puts')
    plt.legend()
    plt.ylabel('Implied volatility (%)')
    plt.xlabel('Moneyness (K/S)')
    plt.title(f'{symbol} volatility skew/smile using yf implied volatilities\n for {dates[exp_index]} options')
    plt.show()

    return calls, puts

#1.2-Calling the function to plot IV skew
calls, puts = pull_option_chain(symbol,7,0.0001,0)


# 2 ---- GBM Montecarlo engine to simulate N amount of stock path over T period  
# 
#                           (0) Manual inputs
window = "1y"   #for vol calculation
rf_ticker = "^IRX"    #risk_free rare used
#                       GBM Montecarlo engine inputs 
Nsim = 1000
T = 1  # Maturity (year fraction)
steps = 252*T

# (2.0)-Fetching other model inputs
S0 = calls['spot_price'].iloc[0]
window = "1y"
#Get historic vol:
# 2.1 - Defining a function that will fetch the symbol's annual volatililty
def pull_hist_vol(symbol,window):
    tk = yf.Ticker(symbol)
    hist = tk.history(period=window)
    hist['log_returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    daily_volatility = hist['log_returns'].std()
    historical_volatility = daily_volatility * np.sqrt(252)
    return daily_volatility, historical_volatility

daily_volatility, historical_volatility = pull_hist_vol(symbol,window)

vol = historical_volatility
# For the exercice of only fetching data from Yahoo Finance, for the risk free rate 
# I Use the 3month t-bill as proxy for the 1Y. Over 1-y the error shouldn't be significant.
# 2.2  -Fetching the data for the selected yf risk-free rate ticker
irx = yf.Ticker(rf_ticker)
current_yield = irx.fast_info['previousClose']
risk_free_rate = current_yield / 100.0
r = risk_free_rate


# 2.3 - Defining the GBM Montecarlo engine for my selected ticker

def simulate_paths(S0,Nsim,steps,r,vol,T):
    paths = np.zeros((Nsim, steps +1))
    paths[:,0] = S0
    dt = T/steps
    r_d =r/T
    vol_d = vol/np.sqrt(T)
    drift_rn = r_d - 0.5*(vol_d)*(vol_d)
    timesteps  = np.arange(0, steps +1)

    for i in range(0, Nsim):
        for j in range(1, steps +1):
            wt = np.random.normal(0,1)*np.sqrt(dt)
            paths[i,j] = paths[i,j-1]*np.exp(drift_rn*(dt) + vol*wt)

    return paths

# 2.4 -  Calling the GBM MC engine function
my_paths = simulate_paths(S0, Nsim, steps, 0.04, vol, T)

# 3.5 Plotting the paths  
plt.plot(my_paths.T)  # since matplotlib reads data vertically, column by column
plt.title(f" {Nsim} simulations of {symbol} price evolution \n over the next {steps} days")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Price")
plt.show()


