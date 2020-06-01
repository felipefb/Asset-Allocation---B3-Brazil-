#!/usr/bin/env python
# coding: utf-8

# In[2]:


import investpy as env
import numpy as np
import pandas as pd


# In[3]:


lt = ['PETR4','VALE3']
prices = pd.DataFrame()
for i in lt:
    df = env.get_stock_historical_data(stock=i, from_date='01/05/2019', to_date='29/05/2020',  country='brazil')
    df['Ativo'] = i
    prices = pd.concat([prices, df], sort=True)


# In[4]:


pivoted = prices.pivot(columns='Ativo', values='Close')
pivoted.head()


# In[5]:


cov_matrix = pivoted.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix


# In[6]:


e_r = pivoted.resample('Y').last().pct_change().mean()
e_r


# In[7]:


sd = pivoted.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
sd


# In[8]:


assets = pd.concat([e_r, sd], axis=1)
assets.columns = ['Returns', 'Volatility']
assets


# In[9]:


p_ret = []
p_vol = []
p_weights = []

num_assets = len(pivoted.columns)
num_portfolios = 1000


# In[10]:


for portfolio in range(num_portfolios):
    #weights = [.25, .75]
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, e_r)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    sd = np.sqrt(var)
    ann_sd = sd*np.sqrt(250)
    p_vol.append(ann_sd)
weights


# In[11]:


data = {'Returns':p_ret, 'Volatility':p_vol}


# In[12]:


for counter, symbol in enumerate(pivoted.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]


# In[13]:


portfolios  = pd.DataFrame(data)
portfolios.head()


# In[14]:


portfolios.plot.scatter(x='Volatility', y='Returns', grid=True)


# In[15]:


op_space = pd.concat([portfolios, assets])
op_space


# In[61]:


op_space.plot.scatter(x='Volatility', y='Returns', grid=True)


# In[ ]:




