#!/usr/bin/env python
# coding: utf-8

import investpy as env
import numpy as np
import pandas as pd


lt = ['PETR4','VALE3']
prices = pd.DataFrame()
for i in lt:
    df = env.get_stock_historical_data(stock=i, from_date='01/05/2019', to_date='29/05/2020',  country='brazil')
    df['Ativo'] = i
    prices = pd.concat([prices, df], sort=True)


pivoted = prices.pivot(columns='Ativo', values='Close')

cov_matrix = pivoted.pct_change().apply(lambda x: np.log(1+x)).cov()

e_r = pivoted.resample('Y').last().pct_change().mean()

sd = pivoted.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

assets = pd.concat([e_r, sd], axis=1)
assets.columns = ['Returns', 'Volatility']

p_ret = []
p_vol = []
p_weights = []

num_assets = len(pivoted.columns)
num_portfolios = 1000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, e_r)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    sd = np.sqrt(var)
    ann_sd = sd*np.sqrt(250)
    p_vol.append(ann_sd)

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(pivoted.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios  = pd.DataFrame(data)

portfolios.plot.scatter(x='Volatility', y='Returns', grid=True)




