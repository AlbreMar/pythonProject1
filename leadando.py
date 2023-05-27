import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression as LM
from sklearn.model_selection import TimeSeriesSplit as TSplit
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from statsmodels.graphics.tsaplots import plot_acf

np.set_printoptions(suppress=True)
gold = pd.read_csv('GLD.csv')
sp500 = pd.read_csv('SPY.csv')

def getReturns1(df1):
    output = pd.DataFrame(index=range(len(df1) - 1), columns=range(1))
    for i in range(1, len(df1)):
        output.iloc[i - 1, 0] = df1['Close'][i] - df1['Close'][i - 1] ##- 1
    return output

def getReturns2(df1, df2):
    output = pd.DataFrame(index=range(len(df1) - 1), columns=range(2))
    for i in range(1, len(df1)):
        output.iloc[i - 1, 0] = df1['Close'][i] / df1['Close'][i - 1] - 1
        output.iloc[i - 1, 1] = df2['Close'][i] / df2['Close'][i - 1] - 1
    return output

sp500Returns = getReturns1(sp500)
returns = getReturns2(gold, sp500)

def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    ewma = df_etf_returns.ewm(alpha = 1 - decay_factor, min_periods = window).var()
    index = ewma.notnull().idxmax()[0]
    ##return ewma
    return ewma.iloc[index : len(ewma),  : ]

sp500ReturnsEwma97 = calculate_ewma_variance(sp500Returns, 0.97, 100)
sp500ReturnsEwma94 = calculate_ewma_variance(sp500Returns, 0.94, 100)

plot.figure(figsize=(10,6))
plot.plot(range(0, len(sp500ReturnsEwma94)), sp500ReturnsEwma94, label='EWMA (decay factor = 0.94)')
plot.plot(range(0, len(sp500ReturnsEwma97)), sp500ReturnsEwma97, label='EWMA (decay factor = 0.97)')
plot.xlabel('Date')
plot.ylabel('Variance')
plot.title('ETF Returns - EWMA Variance')
plot.legend()
plot.show()

sp500ReturnsSq = pd.DataFrame()
sp500ReturnsSq['Diff'] = sp500Returns
sp500ReturnsSq['Sq'] = sp500ReturnsSq['Diff'] ** 2

def pred(data, lag):
    data = data.Sq
    data = pd.DataFrame(data)
    for currentLag in range(1, lag + 1):
        data[f'Lag_{currentLag}'] = data['Sq'].shift(currentLag)

    data.dropna(inplace = True)
    X = data.iloc[ : , -(lag) : ]
    y = data['Sq']


    model = LM()
    dataSplit = TSplit(n_splits = 10)
    mse = []
    r2 = []

    for train, test in dataSplit.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        currentR2 = R2(y_test, y_pred)
        r2.append(currentR2)
        currentMse = MSE(y_test, y_pred)
        mse.append(currentMse)
    return np.mean(r2)
    ##return np.mean(mse)

performance = []
for i in range(1, 20):
    test = pred(sp500ReturnsSq, i)
    performance.append(test)

plot.figure(figsize=(10,6))
plot.plot(range(0, len(performance)), performance, label='Model Performance With Different Lags')
plot.xlabel('Lags')
plot.ylabel('R2')
plot.title('Model Performance With Different Lags')
plot.legend()
plot.show()