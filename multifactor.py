#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:12:32 2020

@author: zhuhx
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import empyrical as ep
import pyfolio as pf
from pypfopt.efficient_frontier import EfficientFrontier

def get_performance_summary(returns):
    stats = {'annualized_returns': ep.annual_return(returns),
             'cumulative_returns': ep.cum_returns_final(returns),
             'annual_volatility': ep.annual_volatility(returns),
             'sharpe_ratio': ep.sharpe_ratio(returns),
             'sortino_ratio': ep.sortino_ratio(returns),
             'max_drawdown': ep.max_drawdown(returns)}
    return pd.Series(stats)

# get financial data from yahoo finance
Stockprice = web.DataReader('^GSPC', 'yahoo','2000-01-01','2019-12-31')
Stockprice.drop(['High','Low','Open','Close','Volume','Adj Close'],axis=1,inplace=True)
raw = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
symbols = raw[0]['Ticker Symbol'].tolist()
symbols.append('^GSPC')
symbols[69] = 'BRK-B'
symbols[81] = 'BF-B'
for symbol in symbols:
    data = web.DataReader(symbol, 'yahoo','2000-01-01','2019-12-31')
    Stockprice[symbol] = data['Adj Close']
Stockreturns = Stockprice.pct_change()
Stockreturns.to_excel('sp500_all.xlsx)

# get Fama French 4 factor data and optimal weights                      
sp505 = pd.read_excel('sp500_all.xlsx') 
ff4 = pd.read_excel('ff4.xlsx')
ff4 = ff4.set_index('date')
mu = ff4.mean()
matrix = ff4.cov()
weights = EfficientFrontier(mu,matrix).max_sharpe(0.)
weight = pd.DataFrame.from_dict(weights,orient = 'index')
weight.to_excel('weights.xlsx')

# run the regression model via WRDS and rank the socre of beta
score = pd.read_excel('scoring.xlsx', sheet_name = 'pivot')
score.drop(index = [313,314],inplace=True)
score['sort_id']=score['scoring'].rank(ascending=False)
score.set_index('sort_id',inplace=True)
score.sort_index(inplace=True)
top50=score['Row Labels'][1:50]
low50=score['Row Labels'][264:313]
top50.to_excel('top50.xlsx')
low50.to_excel('low50.xlsx')

# backtesting
plt.figure(figsize=(20,8))
font = {'family': 'serif',
       'color':  'darkred',
       'weight': 'normal',
       'fontsize': 24}
plt.plot(sp505['Date'],((1+sp505['^GSPC']).cumprod()-1))
plt.plot(sp505['Date'],((1+sp505['Portfolio_ew']).cumprod()-1)/)
plt.plot(sp505['Date'],((1+sp505['Portfolio_score']).cumprod()-1))
plt.legend(['SP 500','Portfolio_ew','Portfolio_score'], loc=2)
plt.ylabel('Cumulative Return',fontsize=16)
plt.xlabel('Day(s)',fontsize=16)
plt.title('Backtesting from 2000 to 2019',fontdict=font)
plt.show()

# performance
get_performance_summary(Stockreturns['^GSPC'])
get_performance_summary(Stockreturns['Portfolio_ew'])
get_performance_summary(Stockreturns['Portfolio_score'])