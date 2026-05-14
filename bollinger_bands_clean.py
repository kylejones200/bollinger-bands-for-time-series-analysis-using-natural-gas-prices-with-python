import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Configuration
np.random.seed(42)

def generate_synthetic_price_data(n_days=365):
    """Generate synthetic stock price data with realistic characteristics."""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Start price and parameters
    base_price = 100
    drift = 0.0005
    volatility = 0.02
    
    # Generate log returns
    returns = np.random.normal(drift, volatility, n_days)
    
    # Convert to price series
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands with proper handling for trading signals.
    
    For visualization: Uses current values (includes today)
    For trading: Uses shifted values (only past data)
    """
    df = df.copy()
    
    # Rolling statistics (includes current value - for visualization)
    df['sma'] = df['price'].rolling(window=window).mean()
    df['std'] = df['price'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (num_std * df['std'])
    df['lower_band'] = df['sma'] - (num_std * df['std'])
    
    # Shifted rolling statistics (excludes current value - for trading signals)
    df['sma_shifted'] = df['price'].shift(1).rolling(window=window).mean()
    df['std_shifted'] = df['price'].shift(1).rolling(window=window).std()
    df['upper_band_shifted'] = df['sma_shifted'] + (num_std * df['std_shifted'])
    df['lower_band_shifted'] = df['sma_shifted'] - (num_std * df['std_shifted'])
    
    return df

def generate_trading_signals(df):
    """
    Generate buy/sell signals based on Bollinger Bands.
    Uses shifted bands to avoid look-ahead bias.
    """
    df = df.copy()
    
    # Initialize signal column (0 = hold, 1 = buy, -1 = sell)
    df['signal'] = 0
    
    # Buy when price crosses below lower band
    df.loc[df['price'] < df['lower_band_shifted'], 'signal'] = 1
    
    # Sell when price crosses above upper band
    df.loc[df['price'] > df['upper_band_shifted'], 'signal'] = -1
    
    # Calculate positions (cumulative signals)
    df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate returns
    df['returns'] = df['price'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    return df

# Generate data
df = generate_synthetic_price_data(n_days=365)

# Calculate Bollinger Bands
df = calculate_bollinger_bands(df, window=20, num_std=2)

# Generate trading signals
df = generate_trading_signals(df)

# Calculate performance metrics
total_return = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100
strategy_cumulative_return = (1 + df['strategy_returns']).cumprod().iloc[-1] - 1
strategy_return = strategy_cumulative_return * 100

# Count signals
buy_signals = (df['signal'] == 1).sum()
sell_signals = (df['signal'] == -1).sum()

logger.info(f"Period: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
logger.info(f"Buy-and-Hold Return: {total_return:.2f}%")
logger.info(f"Bollinger Band Strategy Return: {strategy_return:.2f}%")
logger.info(f"Buy Signals: {buy_signals} | Sell Signals: {sell_signals}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: Price with Bollinger Bands
axes[0].plot(df['date'], df['price'], color='#2c3e50', linewidth=1.5, label='Price')
axes[0].plot(df['date'], df['sma'], color='#3498db', linewidth=1.5, alpha=0.7, label='20-Day SMA')
axes[0].fill_between(df['date'], df['lower_band'], df['upper_band'], 
                      color='#3498db', alpha=0.2, label='Bollinger Bands (±2σ)')
axes[0].plot(df['date'], df['upper_band'], color='#3498db', linewidth=1, alpha=0.5, linestyle='--')
axes[0].plot(df['date'], df['lower_band'], color='#3498db', linewidth=1, alpha=0.5, linestyle='--')
axes[0].set_title('Bollinger Bands (20-Day Window, ±2σ)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='upper left')
axes[0].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Panel 2: Price with Trading Signals
axes[1].plot(df['date'], df['price'], color='#2c3e50', linewidth=1.5, label='Price')
axes[1].fill_between(df['date'], df['lower_band'], df['upper_band'], 
                      color='#95a5a6', alpha=0.1)

# Mark buy signals
buy_dates = df[df['signal'] == 1]['date']
buy_prices = df[df['signal'] == 1]['price']
axes[1].scatter(buy_dates, buy_prices, color='#27ae60', s=80, marker='^', 
               zorder=5, label='Buy Signal', alpha=0.8)

# Mark sell signals
sell_dates = df[df['signal'] == -1]['date']
sell_prices = df[df['signal'] == -1]['price']
axes[1].scatter(sell_dates, sell_prices, color='#e74c3c', s=80, marker='v', 
               zorder=5, label='Sell Signal', alpha=0.8)

axes[1].set_title('Trading Signals', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Price ($)')
axes[1].legend(loc='upper left')
axes[1].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Panel 3: Cumulative Returns Comparison
df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1

axes[2].plot(df['date'], df['cumulative_returns'] * 100, 
            color='#95a5a6', linewidth=2, label='Buy & Hold', alpha=0.7)
axes[2].plot(df['date'], df['cumulative_strategy_returns'] * 100, 
            color='#3498db', linewidth=2, label='Bollinger Strategy')
axes[2].axhline(0, color='black', linewidth=0.5, alpha=0.3)
axes[2].set_title('Strategy Performance Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Cumulative Return (%)')
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left')
axes[2].xaxis.set_major_formatter(DateFormatter('%b %Y'))

plt.tight_layout()
plt.savefig('bollinger_bands_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Band width analysis
fig, ax = plt.subplots(figsize=(14, 4))
df['band_width'] = df['upper_band'] - df['lower_band']
df['band_width_pct'] = (df['band_width'] / df['sma']) * 100

ax.fill_between(df['date'], 0, df['band_width_pct'], color='#3498db', alpha=0.5)
ax.plot(df['date'], df['band_width_pct'], color='#3498db', linewidth=1.5)
ax.set_title('Bollinger Band Width (Volatility Indicator)', fontsize=12, fontweight='bold')
ax.set_ylabel('Band Width (% of SMA)')
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

plt.tight_layout()
plt.savefig('bollinger_band_width.png', dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"\nSaved: bollinger_bands_analysis.png")
logger.info(f"Saved: bollinger_band_width.png")

