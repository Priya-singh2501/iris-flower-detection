#!/usr/bin/env python
# coding: utf-8

# # IMPORT MODULES
# 

# In[83]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[84]:


df = pd.read_csv(r"C:\Users\hp\Desktop\Iris.csv")
df.head(8)


# In[85]:


df.tail(8)


# In[86]:


df.describe()


# In[87]:


df.info()


# In[88]:


df['Species'].value_counts()


# # PREPROCESSING THE DATA SET

# In[89]:


df.isnull().sum()


# # EDA
# 

# In[90]:


df['SepalWidthCm'].hist()


# In[91]:


df['SepalLengthCm'].hist()


# In[92]:


df['PetalLengthCm'].hist()


# In[93]:


df['PetalWidthCm'].hist()


# In[94]:


colors = ['red', 'green','purple']
sepcies = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']    


# In[95]:


species=['Iris-setosa','Iris-versicolor', 'Iris-virginica']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i] , label=species[i])


# In[96]:


species=['Iris-setosa','Iris-versicolor', 'Iris-virginica']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i] , label=species[i])


# In[97]:


species=['Iris-setosa','Iris-versicolor', 'Iris-virginica']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i] , label=species[i])


# In[98]:


species=['Iris-setosa','Iris-versicolor', 'Iris-virginica']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i] , label=species[i])


# # COORELATION MATRIX

# In[99]:


df.corr(numeric_only = True)


# In[100]:


corr = df.corr(numeric_only = True)
fig , ax = plt.subplots(figsize = (10,8))
sns.heatmap(corr , annot = True, ax=ax)


# # LABEL ENCODER

# In[101]:


pip install scikit-learn


# In[102]:


from sklearn.preprocessing import LabelEncoder
le =  LabelEncoder()


# In[103]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# # MODEL TRAINING

# In[113]:


from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
Y = df['Species']
x_train, x_test , y_train ,y_test = train_test_split(X, Y, test_size = 0.30)


# In[114]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[115]:


model.fit(x_train, y_train)


# In[116]:


print("Accuracy:",model.score(x_test , y_test)*100)


# In[108]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[109]:


model.fit(x_train, y_train)


# In[110]:


print("Accuracy:",model.score(x_test , y_test)*100)


# In[111]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[112]:


model.fit(x_train, y_train)


# In[67]:


print("Accuracy:",model.score(x_test , y_test)*100)


# In[ ]:




