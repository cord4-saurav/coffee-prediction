#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df_paths =['~/Desktop/Tea and coffe prediction/archive/domestic-consumption.csv',
         '~/Desktop/Tea and coffe prediction/archive/exports-calendar-year.csv',
          '~/Desktop/Tea and coffe prediction/archive/exports-crop-year.csv',
          '~/Desktop/Tea and coffe prediction/archive/gross-opening-stocks.csv',
          '~/Desktop/Tea and coffe prediction/archive/total-production.csv']


# In[3]:


dfs = [pd.read_csv(df_path) for df_path in df_paths]


# In[4]:


dfs[0].mean(axis=1)


# In[5]:


def get_means(df):
    df = df.copy()
    countries = df[df.columns[0]]
    means = df.mean(axis=1)
    df = pd.concat([countries, means], axis=1)
    df.columns = ['country', countries.name]
    return df


# In[6]:


get_means(dfs[0]).merge(get_means(dfs[1]), on='country')


# In[7]:


def make_df(dfs):
    processesed_dfs = []

    for df in dfs:
        processesed_dfs.append(get_means(df))

    df = processesed_dfs[0]

    for i in range(1, len(processesed_dfs)):
        df = df.merge(processesed_dfs[1], on='country')

    return df


# In[8]:


data = make_df(dfs)
data


# In[12]:


def preprocess_input(df):
    df = df.copy()
    
    y = df['domestic_consumption']
    x = df.drop('domestic_consumption', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=1)
    return  x_train, x_test, y_train, y_test


# In[13]:


x_train, x_test, y_train, y_test = preprocess_input(data)


# In[14]:


x_train


# In[ ]:




