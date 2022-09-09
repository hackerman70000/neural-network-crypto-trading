import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import tensorflow as tf
# data downloaded from https://www.cryptodatadownload.com/data/binance/
df = pd.read_csv('data_binance.csv')

df.rename(columns = {'Volume BTC':'volume'}, inplace = True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by="date")
df.reset_index(drop=True, inplace=True)

# deleting unused data
df.drop('date', inplace=True, axis=1)
df.drop('tradecount', inplace=True, axis=1)
df.drop('unix', inplace=True, axis=1)
df.drop('symbol', inplace=True, axis=1)
df.drop('Volume USDT', inplace=True, axis=1)


df.dropna(inplace=True)

# calculating technical analisis indicators
df['rsi_14'] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
df['roc_12'] = ta.momentum.ROCIndicator(close=df["close"], window=12).roc()
df['roc_20'] = ta.momentum.ROCIndicator(close=df["close"], window=20).roc()
df['srsik_20'] = ta.momentum.StochRSIIndicator(close=df["close"], window=20, smooth1=3, smooth2=3).stochrsi_k()
df['tsi5_10'] = ta.momentum.TSIIndicator(close=df["close"], window_slow=10, window_fast=5).tsi()
df['tsi5_13'] = ta.momentum.TSIIndicator(close=df["close"], window_slow=13, window_fast=5).tsi()
df['bpb'] = ta.volatility.bollinger_pband(close=df["close"] , window=20, window_dev=2)
df['blb'] = ((df['high'] + df['low'])/2)/ ta.volatility.BollingerBands(close=df["close"], window=20).bollinger_lband()
df['ppo26_12'] = ta.momentum.PercentagePriceOscillator(close=df["close"], window_slow=26, window_fast=12, window_sign= 9).ppo()
df['ppo10_50'] = ta.momentum.PercentagePriceOscillator(close=df["close"], window_slow=50, window_fast=10, window_sign= 9).ppo()
df['ppo13_49'] = ta.momentum.PercentagePriceOscillator(close=df["close"], window_slow=49, window_fast=13, window_sign= 9).ppo()
df['ppo3_7'] = ta.momentum.PercentagePriceOscillator(close=df["close"], window_slow=7, window_fast=3, window_sign= 9).ppo()
df['uli_50'] = ta.volatility.UlcerIndex(close=df["open"], window=50).ulcer_index()
df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume = df["volume"], window = 20).chaikin_money_flow() * 100
df['sto'] = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window = 14, smooth_window = 3).stoch()
df['uo'] = ta.momentum.UltimateOscillator(high=df["high"], low=df["low"], close=df["close"], window1 = 7, window2 = 14, window3 = 28, weight1 = 4.0, weight2 = 2.0, weight3 = 1.0).ultimate_oscillator()
df['cci'] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window = 20, constant = 0.015).cci()
df['atr'] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window = 14).average_true_range()

df.dropna(inplace=True)

# creating target values

df['exp']  = np.where(  (df['open'] * 1.051 <= df['high'].shift(-1))
                      | (df['open'] * 1.051 <= df['high'].shift(-2))
                      | (df['open'] * 1.051 <= df['high'].shift(-3))
                      | (df['open'] * 1.051 <= df['high'].shift(-4))
                      | (df['open'] * 1.051 <= df['high'].shift(-5))
                      | (df['open'] * 1.051 <= df['high'].shift(-6))
                       ,1 ,0)


print(df['exp'].value_counts())
df.drop(df.tail(7).index,inplace=True) #do wywalenia
df.drop('high', inplace=True, axis=1)
df.drop('open', inplace=True, axis=1)
df.drop('volume', inplace=True, axis=1)
df.drop('close', inplace=True, axis=1)
df.drop('low', inplace=True, axis=1)
df.dropna(inplace=True)

df = df.iloc[df.shape[0]- 720: , :]
df_train = df.iloc[:df.shape[0]- 100,:]
df_pred =  df.iloc[df.shape[0]- 100:,:]

# splitting into dataset and validation sets
x = df_train.iloc[:, :len(df_train.columns) - 1]
y = df_train['exp']

x_test2 = df_pred.iloc[:, :len(df_pred.columns) - 1]
y_test2 = df_pred['exp']
x_test2.reset_index(drop=True, inplace=True)
y_test2.reset_index(drop=True, inplace=True)

# checking if nan occurs
print(x.isnull().any().any())
print(y.isnull().any().any())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,shuffle = True )

x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# skalowanie danych
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_test2_scaled = scaler.transform(x_test2)

ds_train=lgbm.Dataset(x_train_scaled, label=y_train)
ds_test=lgbm.Dataset(x_test_scaled, label=y_test)


params = {
'boosting_type': 'dart',
'objective': 'binary',
'metric': 'binary_logloss',
'num_class':1,
'min_data_in_leaf':30,
'bagging_fraction':1,
'bagging_freq':10,
'max_depth':8,
'num_leaves':63,
'learning_rate':0.04,
'num_iterations':100,
'extra_trees':'+',
'max_bin':200,
'seed':2
}


model=lgbm.train(params,ds_train)

y_pred = model.predict(x_test2_scaled)

print(y_test2.value_counts())
predictions = pd.DataFrame(y_pred,columns=['0-1'])
predictions['target'] = y_test2


predictions['predictions']  = np.where(predictions['0-1'] > 0.55,1 ,0)


pd.set_option('display.max_rows', None)
print(predictions)

m = tf.keras.metrics.Precision()
m.update_state(predictions['target'], predictions['predictions'])
print(m.result().numpy())
print(predictions['predictions'].sum())
