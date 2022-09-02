import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#from sklearn.metrics import roc_auc_score

# data downloaded from https://www.cryptodatadownload.com/data/binance/
df = pd.read_csv('data_binance.csv')

df.rename(columns = {'Volume BTC':'volume'}, inplace = True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by="date")

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
df['uli_30'] = ta.volatility.UlcerIndex(close=df["open"], window=30).ulcer_index()
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
                       ,1 ,0)



df.drop(df.tail(6).index,inplace=True)
df.drop('high', inplace=True, axis=1)
df.drop('open', inplace=True, axis=1)
df.drop('volume', inplace=True, axis=1)
df.drop('close', inplace=True, axis=1)
df.drop('low', inplace=True, axis=1)
df.dropna(inplace=True)


# splitting into dataset and validation sets
x = df.iloc[:, :len(df.columns) - 1]
y = df['exp']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle = True, random_state = 0)


#lgbm
model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=100, n_jobs=-1, num_leaves=63, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

model.fit(x_train, y_train)


#Feature Importance plot
fig, ax = plt.subplots(figsize=(14,20))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.title("LightGBM - Feature Importance", fontsize=16)
plt.show()


y_pred=model.predict(x_test)

accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print('Test set score: {:.4f}'.format(model.score(x_test, y_test)))

y_pred_train = model.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print(classification_report(y_test, y_pred))