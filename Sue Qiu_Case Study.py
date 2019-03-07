
# coding: utf-8

# # Part I
# Background 
# You are given the Data_Files excel sheet which contains information about potatoes, the company's clients, and a snapshot of client potato positions for a period of time. A quantity of Null represents no position. Management has some questions regarding these data and would like to know your interpretations. You will present your findings at the quarterly management meeting

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[129]:


#read data
data=pd.ExcelFile("/Users/suedy/Downloads/2_Data_Files.xlsx")
potato_info=pd.read_excel(data,"Potatoes_Info")
client_info=pd.read_excel(data,"Client_Info")
position=pd.read_excel(data,"Potatoes_Positions")


# In[12]:


potato_info.head()


# In[13]:


client_info.head()


# In[14]:


position.head()


# # Which clients have the largest potato stockpile based on market value? 
# 1. Greate new column "Market Value" to show market value (price*quantity) for each record
# 2. Greate new column "Daily_total_value_client" to show the daily total stockpile per each customer per day
# 3. Find out the client who get the highest market value per day
# 4. Based on the result, client 82, 27 and 38 have the largest value, and especially client 38 has largest value in marjority days
# 5. Give out client information based on the client id

# In[130]:


position['Quantity'].replace('[NULL]',0,inplace = True) #replace all null in Quantity with 0
position["Market Value"]=position["Price"]*position["Quantity"]
position["Daily_total_value_client"]=position.groupby(["Date","Client ID"])['Market Value'].transform(sum)


# In[131]:


client_largest_stockpile=list(position.loc[position.groupby(["Date"])['Daily_total_value_client'].idxmax(),"Client ID"])
for client in list(set(client_largest_stockpile)):
    print ("Client ID {} : {}".format(client,client_largest_stockpile.count(client)))


# In[132]:


client_info[client_info["Client ID"].isin(list(set(client_largest_stockpile)))]


# # Which clients are the most active?
# 1. Create new column "count_client", which count the number of records per customer per day
# 2. Find out the client with highest count (represent active) per day
# 3. Based on the result, Client 38, 50 are most active, especailly 38 has been active for most days
# 4. Give out client information based on the client id

# In[133]:


position["count_client"]=position.groupby(["Date","Client ID"])["Product ID"].transform("count")
client_active=list(position.loc[position.groupby(["Date"])['count_client'].idxmax(),"Client ID"])
for client in list(set(client_active)):
    print ("Client ID {} : {}".format(client,client_active.count(client)))


# In[135]:


client_info[client_info["Client ID"].isin(list(set(client_active)))]


# # Which potatoes are most activity traded?
# 1. Create new column "count_potato", which count the number of records per product per day
# 2. Find out the potato with highest count (represent active) per day
# 3. Based on the result, Product 16, 172, 45 are most active, especailly 16 has been active for most days
# 4. Give out potato information based on the product id

# In[136]:


position["count_potato"]=position.groupby(["Date","Product ID"])["Client ID"].transform("count")
potato_active=list(position.loc[position.groupby(["Date"])['count_potato'].idxmax(),"Product ID"])
for potato in list(set(potato_active)):
    print ("Product ID {} : {}".format(potato,potato_active.count(potato)))


# In[137]:


potato_info[potato_info["Product ID"].isin(list(set(potato_active)))]


# # What client activity trends do you see?
# 1. the number of activity per client each day
# <br>
# There are four types of client activity trends
# <br>
# Type1: steady and highly active (Client 38)
# <br>
# Type2: unsteady and lowly active (Client 50,27)
# <br>
# Type3: steady and lowly active (Most Client)
# 2. the daily total market value per client each day
# <br>
# Type1: steady market value
# <br>
# Type2: unsteady market value with extreme market value(Client 27)
# 3. the average market value based on clients per day
# <br>
# The type2 clients in second point influence the average market value greatly
# 4. the average number of activity based on clients per day
# <br>
# May-Jun: low volatility
# <br>
# Jun-Aug: high volatility
# <br>
# Aug-Sep: downtrend

# In[138]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Client ID']).count()['Market Value'].unstack().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Number of activity')


# In[139]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Client ID'])['Daily_total_value_client'].mean().unstack().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Number of activity')


# In[140]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Client ID'])['Daily_total_value_client'].mean().groupby('Date').mean().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Average market value / customer')


# In[141]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Client ID'])['Daily_total_value_client'].count().groupby('Date').mean().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Average activity number / customer')


# # What potato price trends do you see?
# 1. the average price per product per day
# <br>
# Basically the trend of price are steady for most potatoes, but some always have high price(Product ID 11,236,227), rest of them are of low price
# 2. the average price of potatoes per day
# <br>
# May-Jun: high volatility
# <br>
# Jun-Aug: low volatility but high average price
# <br>
# Aug-Sep: low price valley

# In[143]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Product ID'])['Price'].mean().unstack().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Potato Price')


# In[29]:


fig, ax = plt.subplots(figsize=(15,7))
position.groupby(['Date','Product ID'])['Price'].mean().groupby('Date').mean().plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Average potato price')


# # Are there any factors that can help predict potato prices?
# 
# 1. Exploratory data analysis
# 2. Data cleaning
# 3. Mixed Model
# 4. Model Evaluation
# 5. Model Interpretation
# <br>
# Linear Regression: R-squared not very high but overall coefficients of most factors in potato_info are significant and can be used to predict potato price 
# <br>
# Linear Mixed Model: in order to further detect if quantity has impact on price, however the result is not significant
# <br>
# Decision Tree: Provide a ranking of factor importance, and quantity is represented as an important factor

# In[144]:


#convert data type to categorical ordinal data
depth={"Very shallow":0,"Very shallow - shallow":1,"Shallow":2,"Shallow - medium":3,
       "Medium":4,"Medium - deep":5,"Deep":6,"Deep - very deep":7,"Very deep":8,"-":4}
maturity={"Very early":0,"Very early to early":1,"Early":2,"First Early":3,"Second Early":4,"Early Maincrop":5,
          "Maincrop":6,"Early to intermediate":7,"Intermediate":8,"Intermediate to late":9,"Late":10,
         "Late Maincrop":11,"Late to very late":12,"Very late":13,"-":6.5}
smoothness={"Rough":0,"Medium":1,"Smooth":2,'-':1}
height={"Very short":0,"Short":1,"Short - medium":2,"Medium":3,"Medium - tall":4,"Tall":5,"Tall - very tall":6,"Very tall":7,"-":3.5}
frequency={"Absent":0,"Few":1,"Medium":2,"Many":3,"Very many":4,'-':2}
fresh={"White":0,"Cream":1,"Light yellow":2,"Medium yellow":3,"Deep yellow":4,"Parti-coloured red":5,"-":2.5}
shape={"Short - oval":0,"Oval":1,"Oval - long":2,"Long":3,"Very long":4,"Round":1.5,"-":2}

potato_info['Depth of eyes']=potato_info['Depth of eyes'].map(depth)
potato_info['Smoothness of skin']=potato_info['Smoothness of skin'].map(smoothness)
potato_info['Maturity']=potato_info['Maturity'].map(maturity)
potato_info['Height of plants']=potato_info['Height of plants'].map(height)
potato_info['Frequency of berries']=potato_info['Frequency of berries'].map(frequency)
potato_info['Shape of tuber']=potato_info['Shape of tuber'].map(shape)
potato_info['Colour of flesh']=potato_info['Colour of flesh'].map(fresh)

for col in ["Country","Colour of skin","Colour of base of lightsprout"]:
    potato_info[col] = potato_info[col].astype('category',copy=False)
for col in list(client_info.columns.values):
    client_info[col] = client_info[col].astype('category',copy=False)
client_info=client_info[['Client ID','Client Type', 'Client Location']]


# In[145]:


#Merge three datasets
data=position.merge(potato_info,how="left",left_on='Product ID',right_on='Product ID').merge(client_info,how="left",left_on='Client ID',right_on='Client ID')
data.drop(["Market Value",'Variety Name','Daily_total_value_client','count_client','count_potato'],axis=1,inplace=True, errors='ignore')


# In[146]:


#Exploratory  Data Analysis
#Summary Data
data.info()
data.describe()
plt.subplot(2, 1, 1)
data.boxplot(['Price'])
plt.subplot(2, 1, 2)
data.boxplot(['Quantity'])
data.isin(['-']).sum(axis=0)/data.shape[0]


# In[147]:


#remove outliers
data=data[data['Price']<=400]
data=data[data["Quantity"]<int(1E9)]


# In[211]:


import statsmodels.api as sm
import scipy.stats as ss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
new=data.drop(["Client ID","Client Type","Client Location"],axis=1)#I think they are uncorrelated with potato price
new.corr()#correlation between numerical data
#new=new.set_index(['Product ID','Date'])


# In[225]:


new_group=new.groupby(['Product ID','Date', 'Price']).agg({'Quantity':sum, 'Country':'first', 
                                                           'Shape of tuber':'first','Colour of skin':'first', 
                                                           'Colour of flesh':'first', 'Depth of eyes':'first',
                                                           'Smoothness of skin':'first', 
                                                           'Colour of base of lightsprout':'first', 
                                                           'Maturity':'first','Height of plants':'first', 
                                                           'Frequency of berries':'first'})
new=new_group.reset_index()


# In[228]:


#Linear Regression 
#create dummy variable
for i in ["Country","Colour of skin","Colour of base of lightsprout"]:
    dummy=pd.get_dummies(new[i])
    new=pd.concat([new,dummy],axis=1)
new=new.drop(["Country","Colour of skin","Colour of base of lightsprout"],axis=1)


# In[230]:


new=new.set_index(['Product ID','Date'])
X = new.iloc[:,1:]
y = new['Price'].apply(np.log)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit() 
y_lr_predict = model.predict(X)
model.summary()


# In[231]:


errors = abs(y_lr_predict  - y)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
meanSquaredError=mean_squared_error(y, y_lr_predict)
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)


# In[234]:


#Linear Mixed Model
import statsmodels.api as sm
import statsmodels.formula.api as smf
new=new.reset_index()
md = smf.mixedlm("Price ~ Quantity", new, groups=new["Product ID"])
mdf = md.fit()
print(mdf.summary())


# In[182]:


#Decision Tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn import utils
from sklearn.metrics import mean_squared_error
from math import sqrt
New=data.drop(['Client ID','Client Type','Client Location'],axis=1)
New=New.set_index('Product ID','Date')
lab_enc = preprocessing.LabelEncoder()
New['Colour of base of lightsprout'] = lab_enc.fit_transform(New['Colour of base of lightsprout'])
New['Colour of skin'] = lab_enc.fit_transform(New['Colour of skin'])
New['Country']=lab_enc.fit_transform(New['Country'])
X=New.iloc[:,2:]
y=New['Price']
encoded = lab_enc.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, encoded, test_size=0.33, random_state=42)
dtc = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
dtc.fit(X_train, y_train)
y_dtc_pred=dtc.predict(X_test) 


# In[172]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[183]:


errors = abs(y_dtc_pred  - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
meanSquaredError=mean_squared_error(y_test, y_dtc_pred)
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
important_feature=pd.DataFrame({"feature":X_test.columns,"importance":dtc.feature_importances_})
important_feature.sort_values("importance",ascending=False,inplace=False).head(10)


# # Are there any factors that can help predict client activity? 
# predict the quantity for certain potato
# <br>
# Decision Tree: Price,Client Type, Client Location

# In[194]:


new1=data[['Date','Client ID','Product ID','Price','Quantity','Client Type','Client Location']]
new1['Product ID']=new1['Product ID'].astype('category')
new1=new1.set_index(['Client ID','Date'])


# In[201]:


X=new1[['Product ID','Price','Client Type','Client Location']]
y=new1['Quantity']
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
new1['Client Type'] = lab_enc.fit_transform(new1['Client Type'])
new1['Client Location'] = lab_enc.fit_transform(new1['Client Location'])
X_train, X_test, y_train, y_test = train_test_split(X, encoded, test_size=0.33, random_state=42)
dtc = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
dtc.fit(X_train, y_train)
y_dtc_pred=dtc.predict(X_test) 


# In[202]:


errors = abs(y_dtc_pred  - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
meanSquaredError=mean_squared_error(y_test, y_dtc_pred)
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
important_feature=pd.DataFrame({"feature":X_test.columns,"importance":dtc.feature_importances_})
important_feature.sort_values("importance",ascending=False,inplace=False).head(10)

