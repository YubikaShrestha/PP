#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


pip install matplotlib


# In[3]:


import pandas as pd


# In[4]:


weather = pd.read_csv("C:\\Users\yubik\Production Project\\okland weather.csv", index_col="DATE")


# In[5]:


weather


# In[6]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[7]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[8]:


core_weather.apply(pd.isnull).sum()


# In[9]:


core_weather["snow"].value_counts()


# In[10]:


core_weather["snow_depth"].value_counts()


# In[11]:


del core_weather["snow"]


# In[12]:


del core_weather["snow_depth"]


# In[13]:


core_weather[pd.isnull(core_weather["precip"])]


# In[14]:


core_weather.loc["2013-12-15",:]


# In[15]:


core_weather["precip"].value_counts() / core_weather.shape[0]


# In[16]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[17]:


core_weather.apply(pd.isnull).sum()


# In[18]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[19]:


core_weather.loc["2011-12-18":"2011-12-28"]


# In[20]:


core_weather = core_weather.fillna(method="ffill")


# In[21]:


core_weather.apply(pd.isnull).sum()


# In[22]:


# Check for missing value defined in data documentation
core_weather.apply(lambda x: (x == 9999).sum())


# In[23]:


core_weather.dtypes


# In[24]:


core_weather.index


# In[25]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[26]:


core_weather.index


# In[27]:


core_weather.index.year


# In[28]:


core_weather[["temp_max", "temp_min"]].plot()


# In[29]:


core_weather.index.year.value_counts().sort_index()


# In[30]:


core_weather["precip"].plot()


# In[31]:


core_weather.groupby(core_weather.index.year).apply(lambda x: x["precip"].sum()).plot()


# In[32]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[33]:


core_weather


# In[34]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[35]:


core_weather


# In[36]:


pip install scikit-learn


# In[37]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)


# In[38]:


predictors = ["precip", "temp_max", "temp_min"]


# In[39]:


train = core_weather.loc[:"2020-12-31"]
test = core_weather.loc["2021-01-01":]


# In[40]:


train


# In[41]:


test


# In[42]:


reg.fit(train[predictors], train["target"])


# In[43]:


predictions = reg.predict(test[predictors])


# In[44]:


from sklearn.metrics import mean_squared_error

mean_squared_error(test["target"], predictions)


# In[45]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[46]:


combined


# In[47]:


combined.plot()


# In[48]:


reg.coef_


# In[49]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()

core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]

core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[50]:


core_weather = core_weather.iloc[30:,:].copy()


# In[51]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_squared_error(test["target"], predictions)
    
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[52]:


predictors = ["precip", "temp_max", "temp_min", "month_day_max", "max_min"]

error, combined = create_predictions(predictors, core_weather, reg)
error


# In[53]:


combined.plot()


# In[60]:


core_weather["monthly_avg"] = core_weather.groupby(core_weather.index.month)["temp_max"].transform(lambda x: x.expanding(1).mean())
core_weather["day_of_year_avg"] = core_weather.groupby(core_weather.index.day_of_year)["temp_max"].transform(lambda x: x.expanding(1).mean())


# In[61]:


error, combined = create_predictions(predictors + ["monthly_avg", "day_of_year_avg"], core_weather, reg)
error


# In[62]:


reg.coef_


# In[63]:


core_weather.corr()["target"]


# In[64]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[65]:


combined.sort_values("diff", ascending=False).head(10)


# In[ ]:




