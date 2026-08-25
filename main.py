
from matplotlib import dates
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.stats import norm


# 0-Pulling option chain for a specific ticker

#Select ticker



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


calls, puts = pull_option_chain("NVDA",7,0.0001,0)

# (1) Montecarlo simulation

S0 = calls['spot_price'].iloc[0]
#Get historic vol:
def pull_hist_vol(symbol):
    tk = yf.Ticker(symbol)
    symbol = symbol
    hist = tk.history(period="1y")
    hist['log_returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    daily_volatility = hist['log_returns'].std()
    historical_volatility = daily_volatility * np.sqrt(252)
    return daily_volatility, historical_volatility, symbol

daily_volatility, historical_volatility, symbol = pull_hist_vol("NVDA")

vol = historical_volatility
# Risk free rate ? Use the 3month t-bill as proxy for the 1Y (To only use yahoo finance)
irx = yf.Ticker("^IRX")
current_yield = irx.fast_info['previousClose']
risk_free_rate = current_yield / 100.0
r = risk_free_rate



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


# Runing
Nsim = 1000
T = 1
steps = 252*T

my_paths = simulate_paths(S0, Nsim, steps, 0.04, vol, T)

# Plotting the paths
plt.plot(my_paths.T) 
plt.title(f" {Nsim} simulations of {symbol} price evolution \n over the next {steps} days")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Price")
plt.show()


