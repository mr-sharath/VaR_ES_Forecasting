import numpy as np
import pandas as pd

def preprocess(ticker):
    df = pd.read_csv(f'data/raw/{ticker}.csv', index_col='Date', parse_dates=True)
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().rename('Returns')
    
    # Split into train/test (80/20)
    train = returns.iloc[:int(0.8 * len(returns))]
    test = returns.iloc[int(0.8 * len(returns)):]
    
    train.to_csv(f'data/splits/{ticker}_train.csv')
    test.to_csv(f'data/splits/{ticker}_test.csv')

preprocess('BLK')