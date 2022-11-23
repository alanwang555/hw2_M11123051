#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split

#picking models for prediction.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

#ensemble models for better performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#error evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

#ignore warning to make notebook prettier
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


#import資料集
path1 = 'C:/dataset/adult.train.csv'
path2 = 'C:/dataset/adult.test.csv'
adult_data_train = pd.read_csv(path1)
adult_data_train.head().T
adult_data_test = pd.read_csv(path2)
adult_data_test.head().T


# In[4]:


#加入訓練集的head
data_header = ['Age','Workclass','fnlwgt','Education','Education-num','Marital_Status','Occupation','Relationship','Race','Sex','Capital-gain','Capital-loss','hrs_per_week','Native-Country','Earning_potential']
adult_data_train = pd.read_csv(path1, names = data_header)
adult_data_train.head()


# In[5]:


#將訓練資料集裡的'?'值拿掉
adult_data_train = adult_data_train.replace(to_replace = '%?%', value = np.nan) #replaces everything with a '?' with Nan
adult_data_train.isna().sum()


# In[6]:


#加入測試集的head
data_header = ['Age','Workclass','fnlwgt','Education','Education-num','Marital_Status','Occupation','Relationship','Race','Sex','Capital-gain','Capital-loss','hrs_per_week','Native-Country','Earning_potential']
adult_data_test = pd.read_csv(path2, names = data_header)
adult_data_test.head()


# In[7]:


#將測試資料集裡的'?'值拿掉
adult_data_test = adult_data_train.replace(to_replace = '%?%', value = np.nan) #replaces everything with a '?' with Nan
adult_data_test.isna().sum()


# In[8]:


all_columns = list(adult_data_train.columns)
print('all_columns:\n {}'.format(all_columns))

categorical_columns = list(adult_data_train.select_dtypes(include=['object']).columns)
print('Categorical columns:\n {}'.format(categorical_columns))

numerical_columns = list(adult_data_train.select_dtypes(include=['int64', 'float64']).columns)
print('Numerical columns:\n {}'.format(numerical_columns))


# In[9]:


#訓練資料集的預處理
null_columns = adult_data_train.columns[adult_data_train.isnull().any()]
adult_data_train[null_columns].isnull().sum()


# In[10]:


for i in list(null_columns):
    adult_data_train[i].fillna(adult_data_train[i].mode().values[0],inplace=True)


# In[11]:


print('{null_sum} \n\n {adult_data_train_info}'.format(null_sum=adult_data_train.isna().sum(), adult_data_train_info=adult_data_train.info()))


# In[12]:


adult_data_train[categorical_columns].head()


# In[13]:


label_encoder = LabelEncoder()
encoded_adult_data_train = adult_data_train
for i in categorical_columns:
    encoded_adult_data_train[i] = label_encoder.fit_transform(adult_data_train[i])
encoded_adult_data_train[categorical_columns].head()


# In[14]:


min_max_scaler = MinMaxScaler()

scaled_encoded_adult_data_train = pd.DataFrame()

column_values = encoded_adult_data_train.columns.values
column_values = column_values[:-1]
print(column_values[-1])

scaled_values = min_max_scaler.fit_transform(encoded_adult_data_train[column_values])

for i in range(len(column_values)):
    scaled_encoded_adult_data_train[column_values[i]] = scaled_values[:,i]
    
scaled_encoded_adult_data_train['hrs_per_week'] = encoded_adult_data_train['hrs_per_week']
scaled_encoded_adult_data_train.sample(10)


# In[15]:


scaled_encoded_adult_data_train.describe().T


# In[16]:


for i in range(len(numerical_columns)):
    plt.figure(figsize=(15,10))
    sns.boxplot(scaled_encoded_adult_data_train[numerical_columns[i]])
plt.show() 


# In[17]:


def outlier_detector(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn,[25,75])
    IQR = Q3 - Q1
    lower_bound = Q1-(1.5*IQR)
    upper_bound = Q3+(1.5*IQR)
    return lower_bound,upper_bound


# In[18]:


lowerbound, upperbound = outlier_detector(scaled_encoded_adult_data_train['Age'])
lowerbound, upperbound


# In[19]:


scaled_encoded_adult_data_train[(scaled_encoded_adult_data_train.Age < lowerbound) | (scaled_encoded_adult_data_train.Age > upperbound)]


# In[20]:


new_columns = numerical_columns.copy()
new_columns.remove('Capital-gain') #Sparse column, must not be treated
new_columns.remove('Capital-loss') #Sparse column, must not be treated
new_columns


# In[21]:


treated_scaled_encoded_adult_data_train = scaled_encoded_adult_data_train.copy()
fig,ax=plt.subplots(figsize=(20,15))
ax=sns.heatmap(treated_scaled_encoded_adult_data_train.corr(),annot=True)


# In[22]:


#測試資料集的預處理
null_columns_test = adult_data_test.columns[adult_data_test.isnull().any()]
adult_data_test[null_columns].isnull().sum()


# In[23]:


for i in list(null_columns_test):
    adult_data_test[i].fillna(adult_data_test[i].mode().values[0],inplace=True)


# In[24]:


print('{null_sum} \n\n {adult_data_test_info}'.format(null_sum=adult_data_test.isna().sum(), adult_data_test_info=adult_data_test.info()))


# In[25]:


adult_data_test[categorical_columns].head()


# In[26]:


encoded_adult_data_test = adult_data_test
for i in categorical_columns:
    encoded_adult_data_test[i] = label_encoder.fit_transform(adult_data_test[i])
encoded_adult_data_test[categorical_columns].head()


# In[27]:


scaled_encoded_adult_data_test = pd.DataFrame()

column_values = encoded_adult_data_test.columns.values
column_values = column_values[:-1]
print(column_values[-1])

scaled_values = min_max_scaler.fit_transform(encoded_adult_data_test[column_values])

for i in range(len(column_values)):
    scaled_encoded_adult_data_test[column_values[i]] = scaled_values[:,i]
    
scaled_encoded_adult_data_test['hrs_per_week'] = encoded_adult_data_test['hrs_per_week']
scaled_encoded_adult_data_test.sample(10)


# In[28]:


scaled_encoded_adult_data_test.describe().T


# In[29]:


for i in range(len(numerical_columns)):
    plt.figure(figsize=(15,10))
    sns.boxplot(scaled_encoded_adult_data_test[numerical_columns[i]])
plt.show() 


# In[30]:


lowerbound, upperbound = outlier_detector(scaled_encoded_adult_data_test['Age'])
lowerbound, upperbound


# In[31]:


scaled_encoded_adult_data_test[(scaled_encoded_adult_data_test.Age < lowerbound) | (scaled_encoded_adult_data_test.Age > upperbound)]


# In[32]:


new_columns = numerical_columns.copy()
new_columns.remove('Capital-gain') #Sparse column, must not be treated
new_columns.remove('Capital-loss') #Sparse column, must not be treated
new_columns


# In[33]:


treated_scaled_encoded_adult_data_test = scaled_encoded_adult_data_test.copy()
fig,ax=plt.subplots(figsize=(20,15))
ax=sns.heatmap(treated_scaled_encoded_adult_data_test.corr(),annot=True)


# In[35]:


print(all_columns)

features = all_columns[:-1]
target = treated_scaled_encoded_adult_data_train['hrs_per_week']
print(features)
print(treated_scaled_encoded_adult_data_train.shape)


# In[36]:


feature_df = treated_scaled_encoded_adult_data_train[features]
print(target.head())
feature_df.head()


# In[37]:


print(all_columns)

features_test = all_columns[:-1]
target_test = treated_scaled_encoded_adult_data_test['hrs_per_week']
print(features_test)
print(treated_scaled_encoded_adult_data_test.shape)


# In[38]:


feature_df_test = treated_scaled_encoded_adult_data_test[features_test]
print(target_test.head())
feature_df_test.head()


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(feature_df, target, test_size=0.2)


# In[40]:


print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)


# In[41]:


#KNN
error_rate = []
# Will take some time
k_values = list(filter(lambda x: x%2==1, range(0,50)))
best_k = 0
for i in k_values:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
print(error_rate.index(np.min(error_rate)))


# In[42]:


plt.figure(figsize=(10,10))
plt.plot(k_values,error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[43]:


knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(x_train, y_train)


# In[44]:


knn_train_score = knn_classifier.score(x_train, y_train)
knn_test_score = knn_classifier.score(x_test, y_test)

print('Train score: {}\nTest score: {}'.format(knn_train_score, knn_test_score))


# In[45]:


knn_prediction = knn_classifier.predict(x_test)

knn_classifier_mae = mean_absolute_error(y_test, knn_prediction)
knn_classifier_rmse = np.sqrt(knn_classifier_mae)
knn_classifier_mape = mean_absolute_percentage_error(y_test, knn_prediction)

print('MAE: {}\nRMSE: {}\nMAPE: {}'.format(knn_classifier_mae, knn_classifier_rmse, knn_classifier_mape))


# In[46]:


#RandomForest
random_forest_classifier = RandomForestClassifier(n_estimators=20, min_samples_split=15, min_impurity_decrease=0.05)
random_forest_classifier.fit(x_train, y_train)


# In[47]:


random_forest_train_score = random_forest_classifier.score(x_train,y_train)
random_forest_test_score = random_forest_classifier.score(x_test,y_test)
print('Train score: {}\nTest score: {}'.format(random_forest_train_score, random_forest_test_score))


# In[48]:


random_forest_prediction = random_forest_classifier.predict(x_test)

random_forest_mae = mean_absolute_error(y_test, random_forest_prediction)
random_forest_rmse = np.sqrt(random_forest_mae)
random_forest_mape = mean_absolute_percentage_error(y_test, random_forest_prediction)

print('MAE: {}\nRMSE: {}\nMAPE: {}'.format(random_forest_mae, random_forest_rmse, random_forest_mape))


# In[49]:


#XGboost
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
xgboostModel = XGBClassifier(n_estimators=20, learning_rate= 0.3)
xgboostModel.fit(x_train, y_train)


# In[50]:


xgboost_train_score = xgboostModel.score(x_train,y_train)
xgboost_test_score = xgboostModel.score(x_test,y_test)
print('Train score: {}\nTest score: {}'.format(xgboost_train_score, xgboost_test_score))


# In[51]:


xgboost_prediction = xgboostModel.predict(x_test)

xgboost_mae = mean_absolute_error(y_test, xgboost_prediction)
xgboost_rmse = np.sqrt(xgboost_mae)
xgboost_mape = mean_absolute_percentage_error(y_test, xgboost_prediction)

print('MAE: {}\nRMSE: {}\nMAPE: {}'.format(xgboost_mae, xgboost_rmse, xgboost_mape))

