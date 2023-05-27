import pandas as pd

gold = pd.read_csv('GLD.csv')
sp500 = pd.read_csv('SPY.csv')

def getReturns(df1, df2):
    output = pd.DataFrame(index=range(len(df1) - 1), columns=range(2))
    for i in range(1, len(df1)):
        output.iloc[i - 1, 0] = df1['Close'][i] - df1['Close'][i - 1]
        output.iloc[i - 1, 1] = df2['Close'][i] - df2['Close'][i - 1]
    return output

returns = getReturns(gold, sp500)



