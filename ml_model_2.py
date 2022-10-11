#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# In[6]:


url = "http://localhost:8000/n_chunks"
response = requests.get(url)
response.json()["chunks"]


# In[17]:


url = "http://localhost:8000/chunk/1"
response1 = requests.get(url)
json_text1 = response1.text


# In[18]:


short1 = json_text1[1:-1]
short1 = short1.replace("\\","")


# In[19]:


url = "http://localhost:8000/chunk/2"
response2 = requests.get(url)
json_text2 = response2.text


# In[20]:


short2 = json_text2[1:-1]
short2 = short2.replace("\\","")


# In[21]:


df1 = pd.read_json(short1, orient="records")
df2 = pd.read_json(short2,orient="records")


# In[32]:


data = pd.concat([df1,df2]).dropna()


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(
        data[["gender"]],
        data["Retention"],
    )


# In[34]:


# Randomforest
clf = RandomForestClassifier(max_depth=2, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[41]:


cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, square=True, annot=True, fmt='d', cmap="Blues")
# plt.xlabel('predicted label')
# plt.ylabel('actual label');

print("random forest classification report:")
print(classification_report(y_test, y_pred))


# In[ ]:




