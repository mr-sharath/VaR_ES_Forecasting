import pandas as pd
import pandas_datareader as pdr

def fetch_data(ticker, start_date, end_date):
    df = pdr.DataReader(ticker, 'stooq').sort_index(ascending=True)
    df = df[['Close']].resample('D').last().ffill()  # Daily closing prices
    df.to_csv(f'data/raw/{ticker}.csv')
    return df

# Example: Fetch BlackRock data
fetch_data('BLK', '2000-01-01', '2023-12-31')