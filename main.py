import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# 1.0-Pulling option chain for a specific ticker

# Setting the ticker & Initial inputs
symbol = input('Enter a Yahoo finance ticker (ex: AAPL): ')
min_iv = 0.0001 #  1bp
min_volume = 0
# 0 Creating a date selector (one Ticker object reused everywhere)
tk = yf.Ticker(symbol)
dates = tk.options
print(f"List of expiry dates for {symbol} options")
print(dict(enumerate(dates)))
# Input for exp_index
exp_index = int(input('Enter the nbr: '))


# 1.1-Defining the function that will fetch the option chain from yf
def pull_option_chain(tk, exp_index, min_iv, min_volume):
    target_chain = tk.option_chain(tk.options[exp_index])
    # Getting the current spot
    spot_price = target_chain.underlying.get('regularMarketPrice')
    cols = ['strike', 'lastPrice', 'impliedVolatility']

    def clean(df):
        # Filtering with one mask, slicing columns, sorting by strike
        mask = (df['volume'] > min_volume) & (df['impliedVolatility'] > min_iv)
        df = df.loc[mask, cols].sort_values('strike')
        # Adding the spot and Moneyness (assign avoids SettingWithCopy warnings)
        return df.assign(spot_price=spot_price,
                         moneyness=df['strike'] / spot_price,
                         bs_logmoneyness=np.log(spot_price / df['strike']))

    calls, puts = clean(target_chain.calls), clean(target_chain.puts)

    #Plotting (IV, moneyness) for Calls and puts
    plt.plot(calls['moneyness'], calls['impliedVolatility']*100, label='Calls')
    plt.plot(puts['moneyness'], puts['impliedVolatility']*100, label='Puts')
    plt.legend()
    plt.ylabel('Implied volatility (%)')
    plt.xlabel('Moneyness (K/S)')
    plt.title(f'{symbol} volatility skew/smile using yf implied volatilities\n for {tk.options[exp_index]} options')
    plt.show()

    return calls, puts, spot_price

#1.2-Calling the function to plot IV skew 
calls, puts, S0 = pull_option_chain(tk, exp_index, min_iv, min_volume)


# 2 ---- GBM Montecarlo engine to simulate N amount of stock path over T period
#
#                           (0) Manual inputs
window = "1y"   #for vol calculation
rf_ticker = "^IRX"    #risk_free rate used
#                       GBM Montecarlo engine inputs
Nsim = 1000
T = 1  # Maturity (year fraction)
steps = 252*T

# 2.1 - Defining a function that will fetch the symbol's annual volatility
def pull_hist_vol(tk, window):
    hist = tk.history(period=window, interval="1d") 
    log_returns = np.log(hist['Close']).diff().dropna()
    daily_volatility = log_returns.std()
    historical_volatility = daily_volatility * np.sqrt(252)
    return daily_volatility, historical_volatility

daily_volatility, vol = pull_hist_vol(tk, window)

# For the exercice of only fetching data from Yahoo Finance, for the risk free rate
# I Use the 3month t-bill as proxy for the 1Y. Over 1-y the error shouldn't be significant.
# 2.2  -Fetching the data for the selected yf risk-free rate ticker
r = yf.Ticker(rf_ticker).fast_info['previousClose'] / 100.0


# 2.3 - Defining the GBM Montecarlo engine. (Chose to make it fully vectorized, and not use python loops)
def simulate_paths(S0, Nsim, steps, r, vol, T):
    dt = T/steps
    drift_rn = r - 0.5*vol*vol
    # All the random shocks at once, then cumulative sum of the log increments
    Z = np.random.standard_normal((Nsim, steps))
    log_incr = drift_rn*dt + vol*np.sqrt(dt)*Z
    log_paths = np.cumsum(log_incr, axis=1)
    # Initializing the initial spot as the first column
    paths = S0 * np.exp(np.hstack([np.zeros((Nsim, 1)), log_paths]))
    return paths

# 2.4 -  Calling the GBM MC engine function 
my_paths = simulate_paths(S0, Nsim, steps, r, vol, T)

# 2.5 Plotting the paths
plt.plot(my_paths.T, linewidth=0.5)  # since matplotlib reads data vertically, column by column
plt.title(f" {Nsim} simulations of {symbol} price evolution \n over the next {steps} days")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Price")
plt.show()
