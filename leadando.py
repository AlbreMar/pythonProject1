import pandas as pd
import numpy as np
import os as os
import sys as sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LM
from sklearn.model_selection import TimeSeriesSplit as TSplit
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from statsmodels.graphics.tsaplots import plot_acf

##1. feladat

def read_etf_file(etf):
    filename = os.path.join(etf + '.csv')
    dataframe = pd.read_csv(filename, index_col=0)
    dataframe.index = pd.to_datetime(dataframe.index)
    return dataframe

def get_etf_returns(etf_name, return_type, fieldname='Adj Close'):

    df = read_etf_file(etf_name)
    df = df[[fieldname]]

    df['shifted'] = df.shift(1)
    if return_type=='log':
        df['return'] = np.log(df[fieldname]/df['shifted'])
    if return_type=='simple':
        df['return'] = df[fieldname]/df['shifted']-1

    df = df[['return']]
    df.columns = [etf_name]
    return df

def get_portfolio_return(d_weights):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = get_etf_returns(etf, return_type='simple')
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    df_weighted_returns = df_joined * pd.Series(d_weights)
    s_portfolio_return = df_weighted_returns.sum(axis=1)
    return pd.DataFrame(s_portfolio_return, columns=['pf'])

def generate_portfolio_returns():
    weights = np.arange(0, 1.0, 0.05)
    returns_data = []
    for weight in weights:
        d_weights = {'SPY': weight, 'GLD': 1 - weight}
        df_returns = get_portfolio_return(d_weights)
        returns_data.append(df_returns)

    df_portfolio_returns = pd.concat(returns_data, axis=1)
    df_portfolio_returns.columns = [f'Weight_{weight:.2f}' for weight in weights]
    return df_portfolio_returns

def plotting():
    df_portfolio_returns = generate_portfolio_returns()
    df_portfolio_returns = df_portfolio_returns[["Weight_0.05", "Weight_0.50", "Weight_0.95"]]
    plt.figure(figsize=(12, 6))
    for column in df_portfolio_returns.columns:
        plt.plot(df_portfolio_returns[column], label=column)

    plt.xlabel('Dátum')
    plt.ylabel('A Portfólio Hozamai')
    plt.title('A Portfólió hozamai különböző súlyok mellett')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_historical_var(df_portfolio_returns, alpha):
    l_quantiles = [1 - alpha]
    df_pf = df_portfolio_returns
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = [alpha]
    return df_result.iloc[0]

df_portfolio_returns = generate_portfolio_returns()
alpha = 0.95
var_values = {}
for column in df_portfolio_returns.columns:
    var_value = calculate_historical_var(df_portfolio_returns[column], alpha)
    var_values[column] = var_value

def plot_var_values(var_values):
    weights = np.arange(0, 1.0, 0.05)
    var_list = [var_values[f'Weight_{weight:.2f}'] for weight in weights]

    plt.figure(figsize=(8, 6))
    plt.plot(weights, var_list, marker='o', linestyle='-', color='blue')

    plt.xlabel('SPY súlya')
    plt.ylabel('VaR')
    plt.title('VaR az SPY különböző súlyai mellett')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#2. feladat

def simulated_returns(expected_return, volatility, correlation, num_of_sim):
    cov_matrix = np.array([[volatility[0]**2, correlation * volatility[0] * volatility[1]],
                           [correlation * volatility[0] * volatility[1], volatility[1]**2]])

    np.random.seed(0)
    simulated_returns = np.random.multivariate_normal(expected_return, cov_matrix, num_of_sim)

    return simulated_returns

df_SPY = get_etf_returns("SPY", "simple")
df_GLD = get_etf_returns("GLD", "simple")
return_SPY = df_SPY["SPY"]
return_GLD = df_GLD["GLD"]
expected_return_SPY = return_SPY.mean()
expected_return_GLD = return_GLD.mean()
vola_SPY = return_SPY.std()
vola_GLD = return_GLD.std()
weight_1 = 0.45
weight_2 = 0.55
total_volatility = vola_SPY + vola_GLD
weight_1 = weight_1 * (vola_GLD / total_volatility)
weight_2 = weight_2 * (vola_SPY / total_volatility)


def plot_simulated_returns(simulated_returns):
    num_of_simulations, num_of_assets = simulated_returns.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_of_assets):
        ax.plot(simulated_returns[:, i], label=f"Eszköz {i+1} hozama")

    ax.set_ylabel("Hozamok")
    ax.legend()

    plt.tight_layout()
    plt.show()


correlation_values = np.arange(-1.0, 1.0, 0.2)
var_values2 = []
def calculate_simulated_return_var(asset_returns, weights, alpha):
    portfolio_returns = np.dot(asset_returns, weights)
    portfolio_var = np.percentile(portfolio_returns, 1-alpha)

    return portfolio_var
num_of_simulations = 500
for correlation in correlation_values:
    simulated_returns_array = simulated_returns([expected_return_SPY, expected_return_GLD],
                                                 [vola_SPY, vola_GLD],
                                                 correlation,
                                                 num_of_simulations)

    portfolio_var = calculate_simulated_return_var(simulated_returns_array, [weight_1, weight_2], 0.95)
    var_values2.append(portfolio_var)

def plotting_correlation():
    plt.plot(correlation_values, var_values2, marker='s', linestyle='-', linewidth=2)
    plt.xlabel('Korreláció')
    plt.ylabel('VaR')
    plt.title('VaR érték különböző korreláció mellett')
    plt.grid(True)
    plt.show()


## 3.feladat
np.set_printoptions(suppress=True)
gold = pd.read_csv('GLD.csv')
sp500 = pd.read_csv('SPY.csv')

def getReturns1(df1):
    output = pd.DataFrame(index=range(len(df1) - 1), columns=range(1))
    for i in range(1, len(df1)):
        output.iloc[i - 1, 0] = df1['Close'][i] / df1['Close'][i - 1] - 1
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
    return ewma.iloc[index : len(ewma),  : ]

sp500ReturnsEwma97 = calculate_ewma_variance(sp500Returns, 0.97, 100)
sp500ReturnsEwma94 = calculate_ewma_variance(sp500Returns, 0.94, 100)

plt.figure(figsize=(10,6))
plt.plot(range(0, len(sp500ReturnsEwma94)), sp500ReturnsEwma94, label='EWMA (factor = 0.94)')
plt.plot(range(0, len(sp500ReturnsEwma97)), sp500ReturnsEwma97, label='EWMA (factor = 0.97)')
plt.xlabel('Idő')
plt.ylabel('Variancia')
plt.title('ETF Hozamok - EWMA Variancia')
plt.legend()
plt.show()

## 4.feladat
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

    for train, test in dataSplit.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        currentMse = MSE(y_test, y_pred)
        mse.append(currentMse)
    return np.mean(mse)

performance = []
for i in range(1, 21):
    test = pred(sp500ReturnsSq, i)
    performance.append(test)

plt.figure(figsize=(10,6))
plt.plot(range(1, len(performance) + 1), performance, 'o', label=None)
plt.xlabel('Késleltetés')
plt.ylabel('MSE')
plt.title('Modell Pontosság Különböző Késleltetésekkel')
plt.legend()
plt.xticks(range(1, len(performance) + 1))
plt.show()

plot_acf(sp500ReturnsSq['Sq'], lags=20)
plt.xlabel('Késleltetés')
plt.ylabel('Autókorreláció')
plt.title('Autókorrelációs Függvény (ACF)')
plt.show()