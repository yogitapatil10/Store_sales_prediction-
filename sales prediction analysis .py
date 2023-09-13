#!/usr/bin/env python
# coding: utf-8

# ## Big-Mart sales prediction

# In[ ]:


The aim is to build a predictive model and find out the sales of each product at a particular store. Create a model by which Big Mart can analyse and predict the outlet production sales.
A perfect project to learn Data Analytics and apply Machine Learning algorithms (Linear Regression, Random Forest Regression) to predict the outlet production sales.


# # Dataset Description

# In[ ]:


Big Mart has collected sales data from the year 2013, for 1559 products across 10 stores in different cities. Where the dataset consists of 12 attributes like Item Fat, Item Type, Item MRP, Outlet Type, Item Visibility, Item Weight, Outlet Identifier, Outlet Size, Outlet Establishment Year, Outlet Location Type, Item Identifier and Item Outlet Sales. Out of these attributes response variable is the Item Outlet Sales attribute and remaining attributes are used as the predictor variables.
The data-set is also based on hypotheses of store level and product level. Where store level involves attributes like:- city, population density, store capacity, location, etc. and the product level hypotheses involves attributes like:- brand, advertisement, promotional offer, etc.


# # Dataset Details
# The data has 8523 rows of 12 variables.
# 

# # Variable - Details
# •Item_Identifier- Unique product ID
# •Item_Weight- Weight of product
# •Item_Fat_Content - Whether the product is low fat or not
# •Item_Visibility - The % of total display area of all products in a store allocated to the particular product
# •Item_Type - The category to which the product belongs
# •Item_MRP - Maximum Retail Price (list price) of the product
# •Outlet_Identifier - Unique store ID
# •Outlet_Establishment_Year- The year in which store was established
# •Outlet_Size - The size of the store in terms of ground area covered
# •Outlet_Location_Type- The type of city in which the store is located
# •Outlet_Type- Whether the outlet is just a grocery store or some sort of supermarket
# •Item_Outlet_Sales - Sales of the product in the particular store. This is the outcome variable to be predicted.
# 

# # Project Flow

# In[ ]:



We will handle this problem in a structured way.
      Loading Packages and Data
•Data Structure and Content
•Exploratory Data Analysis
•Missing Value Treatment
•Feature Engineering
•Encoding Categorical Variables
•Label Encoding
•PreProcessing Data
•Modeling
•Linear Regression
•Random Forest Regression


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:\\Users\\\BHUMII\\Downloads\\Train.csv')
df.head()


# In[3]:


# statistical info
df.describe()


# In[4]:


#data type of attributes
df.info()


# In[5]:


# check unique value in dataset
df.apply(lambda x:len(x.unique()))


# ##preprocessing the dataset

# In[6]:


# check for null values
df.isnull().sum()


# In[7]:


# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col


# In[8]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col


# In[ ]:


# print the categorical column
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


# In[12]:


# fill the missing values
Item_Weight_mean = df.pivot_table(values = 'Item_Weight',index = 'Outlet_Identifier')
Item_Weight_mean                                 


# In[13]:


df.columns


# In[14]:


miss_bool = df['Item_Weight'].isnull()
miss_bool


# In[15]:


for i,item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in Item_Weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
             df['Item_Weight'][i] = np.mean(df['Item_Weight'])


# In[16]:


df['Item_Weight'].isnull().sum()


# In[17]:


outlet_size_mode = df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))
outlet_size_mode


# In[18]:


miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[19]:


df['Outlet_Size'].isnull().sum()


# In[20]:


#replace zeros with mean
df.loc[:,'Item_Visibility'].replace([0],[df['Item_Visibility'].mean()],inplace=True)


# In[21]:


sum(df['Item_Visibility']==0)


# In[22]:


# combine item for content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()


# ## create new attributes

# In[23]:


df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']


# In[24]:


df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df['New_Item_Type'].value_counts()


# In[25]:


df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()


# In[26]:


#create small values for establishment year
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']


# In[27]:


df['Outlet_Years']


# In[28]:


df.head()


# In[29]:


sns.distplot(df['Item_Weight'])


# In[30]:


sns.distplot(df['Item_Visibility'])


# In[31]:


sns.distplot(df['Item_MRP'])


# In[32]:


sns.distplot(df['Item_Outlet_Sales'])


# In[33]:


#log transformation
df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])


# In[34]:


sns.distplot(df['Item_Outlet_Sales'])


# In[35]:


sns.countplot(df["Item_Fat_Content"])


# In[36]:


#plt.figure(figsize=(15,5))
l = list(df['Item_Type'].unique())
chart = sns.countplot(df["Item_Type"])
chart.set_xticklabels(labels=l,rotation=90)


# In[37]:


sns.countplot(df['Outlet_Establishment_Year'])


# In[38]:


sns.countplot(df['Outlet_Size'])


# In[39]:


sns.countplot(df['Outlet_Location_Type'])


# In[40]:


sns.countplot(df['Outlet_Type'])


# ##Coorelation matrix

# In[41]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# # Lable Encoding

# In[42]:


df.head()


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content','Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])


# In[44]:


df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])


# In[45]:


df.head()


# ## model trainig

# In[46]:


x = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']


# In[47]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model, x, y):
    # train the model
    model.fit(x,y)

    # predict the training set
    pred = model.predict(x)
    
    # perform a cross validation
    cv_score = cross_val_score(model,x,y,scoring='neg_mean_squared_error')
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV score:", cv_score)


# In[48]:


from sklearn.linear_model import LinearRegression, Ridge,Lasso
model = LinearRegression(normalize=True)
train(model,x,y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[49]:


model = Ridge(normalize=True)
train(model,x,y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[50]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model,x,y)
coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# In[51]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model,x,y)
coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# In[ ]:





# In[ ]:





# In[ ]:




