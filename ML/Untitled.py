#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('housing.csv.gz')


# In[2]:


df.head(10)


# In[3]:


df.info()


# In[4]:


df.ocean_proximity.value_counts()


# In[5]:


df.ocean_proximity.describe()


# In[6]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[7]:


df.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[8]:


df.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4, figsize=(7,3), colorbar=True,s=df["population"]/100, label="population",c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[9]:


df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik_korelacji"})


# In[10]:


import seaborn as sns
sns.pairplot(df)


# In[11]:


from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set),len(test_set)


# In[12]:


train_set.corr().to_pickle('train_set.pkl')


# In[13]:


test_set.corr().to_pickle('test_set.pkl')


# In[ ]:




