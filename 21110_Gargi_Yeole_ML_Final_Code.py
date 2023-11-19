#!/usr/bin/env python
# coding: utf-8

# # Importing some necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# # Data Collection and Pre-Processing

# In[2]:


#Loading the datasets
test_data = pd.read_csv('test_data.csv')
train_labels = pd.read_csv("training_data_targets.csv")
train_data= pd.read_csv('training_data.csv')


# In[3]:


#Inspecting the first 5 rows of the dataframes
test_data.head()


# In[4]:


#Inspecting the first 5 rows of the dataframes
test_data.head()


# In[5]:


#Checking numberof rows and columns in the dataframes
test_data.shape


# In[6]:


train_data.shape


# In[7]:


test_data.columns


# In[8]:


train_data.columns


# # Getting some information about the dataset

# In[9]:


#getting some information about the dataset
test_data.describe()


# In[10]:


train_data.describe()


# In[11]:


test_data.info()


# In[12]:


train_data.info()


# # Checking the missing values

# In[13]:


train_data.isnull().sum()


# In[14]:


test_data.isnull().sum()


# # Imputing the missing values

# In[15]:


# Impute missing values with the mean
median_seats = train_data['Seats'].median()
train_data['Seats'].fillna(median_seats, inplace=True)


# In[16]:


# Assuming 'Mileage' is the column with missing values and it's given in kmpl
mean_mileage = train_data['Mileage'].str.extract('(\d+\.\d+)').astype(float).mean()

# Impute missing values with the mean
train_data['Mileage'].fillna(f'{mean_mileage} kmpl', inplace=True)


# In[17]:


# Assuming 'Engine' is the column with missing values and it's given in CC
mean_engine = train_data['Engine'].str.extract('(\d+\.\d+)').astype(float).mean()

# Impute missing values with the mean
train_data['Engine'].fillna(f'{mean_engine} cc', inplace=True)


# In[18]:


# Assuming 'Power' is the column with missing values and it's given in bhp
mean_power = train_data['Mileage'].str.extract('(\d+\.\d+)').astype(float).mean()

# Impute missing values with the mean
train_data['Power'].fillna(f'{mean_power} bhp', inplace=True)


# In[19]:


train_data.isnull().sum()


# In[20]:


#Imputing the missing values with the median
median_seats = test_data['Seats'].median()
test_data['Seats'].fillna(median_seats, inplace=True)


# In[21]:


# Assuming 'Mileage' is the column with missing values and it's given in kmpl
mean_mileage = test_data['Mileage'].str.extract('(\d+\.\d+)').astype(float).mean()

# Impute missing values with the mean
test_data['Mileage'].fillna(f'{mean_mileage} kmpl', inplace=True)


# In[22]:


# Assuming 'Engine' is the column with missing values and it's given in CC
mean_engine = test_data['Engine'].str.extract('(\d+\.\d+)').astype(float).mean()
# Impute missing values with the mean
test_data['Engine'].fillna(f'{mean_engine} cc', inplace=True)


# In[23]:


# Assuming 'Power' is the column with missing values and it's given in bhp
mean_power = test_data['Power'].str.extract('(\d+\.\d+)').astype(float).mean()

# Impute missing values with the mean
test_data['Power'].fillna(f'{mean_power} bhp', inplace=True)


# In[24]:


test_data.isnull().sum()


# # Checking the categorical data

# In[25]:


print(train_data.Location.value_counts())
print(train_data.Fuel_Type.value_counts())
print(train_data.Transmission.value_counts())
print(train_data.Owner_Type.value_counts())


# In[26]:


print(test_data.Location.value_counts())
print(test_data.Fuel_Type.value_counts())
print(test_data.Transmission.value_counts())
print(test_data.Owner_Type.value_counts())


# In[27]:


#Label Encoding "FUEL_TYPE" Column
#train_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}}, inplace= True)

#Label Encoding "TRANSMISSION" Column
#train_data.replace({'Transmission':{'Manual':0,'Automatic':1,}}, inplace= True)

#Label Encoding "OWNER_TYPE" Column
#train_data.replace({'Owner_Type':{'First':0,'Second':1,'Third':2,'Fourth & Above':3}}, inplace= True)

#Label Encoding "LOCATION" Column
#train_data.replace({'Location':{'Mumbai':0,'Hyderabad':1,'Coimbatore':2,'Kochi':3,'Pune':4,'Delhi':5,'Kolkata':6,'Chennai':7,'Jaipur':8,'Bangalore':9,'Ahmedabad':10}}, inplace= True)


# In[28]:


train_data.head()


# In[29]:


#Label Encoding "FUEL_TYPE" Column
#test_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}}, inplace= True)

#Label Encoding "TRANSMISSION" Column
#test_data.replace({'Transmission':{'Manual':0,'Automatic':1,}}, inplace= True)

#Label Encoding "OWNER_TYPE" Column
#test_data.replace({'Owner_Type':{'First':0,'Second':1,'Third':2,'Fourth & Above':3}}, inplace= True)

#Label Encoding "LOCATION" Column
#test_data.replace({'Location':{'Mumbai':0,'Hyderabad':1,'Coimbatore':2,'Kochi':3,'Pune':4,'Delhi':5,'Kolkata':6,'Chennai':7,'Jaipur':8,'Bangalore':9,'Ahmedabad':10}}, inplace= True)


# In[30]:


test_data.head()


# # Encoding of the dataset

# In[31]:


#from sklearn.preprocessing import LabelEncoder

#label_encoder = LabelEncoder()

# Fit and transform 'Brand' column to numerical labels for "Testing dataset"
#test_data['Brand_Label'] = label_encoder.fit_transform(test_data['Brand'])

#print(test_data)


# In[32]:


# Fit and transform 'Brand' column to numerical labels for "Training Dataset"
#train_data['Brand_Label'] = label_encoder.fit_transform(train_data['Brand'])

#print(train_data)


# # One Hot Encoding

# In[33]:


# One-hot encoding for "Fuel_Type" Column
test_data = pd.get_dummies(test_data, columns=['Fuel_Type'], prefix='Fuel_Type')

# One-hot encoding for "Transmission" Column
test_data = pd.get_dummies(test_data, columns=['Transmission'], prefix='Transmission')

# One-hot encoding for "Owner_Type" Column
test_data = pd.get_dummies(test_data, columns=['Owner_Type'], prefix='Owner_Type')

# One-hot encoding for "Fuel_Type" Column
train_data = pd.get_dummies(train_data, columns=['Fuel_Type'], prefix='Fuel_Type')

# One-hot encoding for "Transmission" Column
train_data = pd.get_dummies(train_data, columns=['Transmission'], prefix='Transmission')

# One-hot encoding for "Owner_Type" Column
train_data = pd.get_dummies(train_data, columns=['Owner_Type'], prefix='Owner_Type')

# Drop columns 'Brand','Location',Year 
columns_to_drop = ['Brand', 'Location', 'Year']
train_data = train_data.drop(columns=columns_to_drop)

columns_to_drop = ['Brand', 'Location', 'Year']
test_data = test_data.drop(columns=columns_to_drop)


# In[34]:


# Extracting the numerical columns and converting them to the appropriate types
train_data['Mileage'] = train_data['Mileage'].str.extract('(\d+\.\d+)').astype(float)
train_data['Engine'] = train_data['Engine'].str.extract('(\d+)').astype(float)
train_data['Power'] = train_data['Power'].str.extract('(\d+\.\d+)').astype(float)
test_data['Mileage'] = test_data['Mileage'].str.extract('(\d+\.\d+)').astype(float)
test_data['Engine'] = test_data['Engine'].str.extract('(\d+)').astype(float)
test_data['Power'] = test_data['Power'].str.extract('(\d+\.\d+)').astype(float)


# In[35]:


test_data


# In[36]:


train_data


# # Correlation Matrix 

# In[37]:


#Finding the correlation matrix for test dataset:
correlation = test_data.corr()
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns)


# In[38]:


#Finding the correlation matrix for train dataset:
correlation = train_data.corr()
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns)


# Scatter Plots:-

# In[39]:


#scatter plot for numerical features
num_features = ["Kilometers_Driven","Mileage","Seats","Engine","Power"]
sns.pairplot(train_data[num_features],size = 3.0)
plt.show();


# In[40]:


#scatter plot for numerical features
num_features = ["Kilometers_Driven","Mileage","Seats","Engine","Power"]
sns.pairplot(test_data[num_features],size = 3.0)
plt.show();


# In[41]:


#scatter plot for categorical features:
num_features = ['Fuel_Type_CNG','Fuel_Type_LPG','Fuel_Type_Electric','Fuel_Type_Diesel','Fuel_Type_Petrol',
       'Transmission_Automatic','Transmission_Manual', 'Owner_Type_First','Owner_Type_Second','Owner_Type_Third','Owner_Type_Fourth & Above']
sns.pairplot(train_data[num_features],size = 3.0)
plt.show();


# In[42]:


#scatter plot for categorical features:
num_features = ['Fuel_Type_CNG','Fuel_Type_LPG','Fuel_Type_Electric','Fuel_Type_Diesel','Fuel_Type_Petrol',
       'Transmission_Automatic','Transmission_Manual', 'Owner_Type_First','Owner_Type_Second','Owner_Type_Third','Owner_Type_Fourth & Above']
sns.pairplot(test_data[num_features],size = 3.0)
plt.show();


# # Linear Regression

# In[43]:


test_data2 = test_data[['Kilometers_Driven', 'Fuel_Type_CNG','Fuel_Type_LPG','Fuel_Type_Electric','Fuel_Type_Diesel','Fuel_Type_Petrol',
       'Transmission_Automatic','Transmission_Manual', 'Owner_Type_First','Owner_Type_Second','Owner_Type_Third','Owner_Type_Fourth & Above', 'Mileage', 'Engine', 'Seats']]


# In[44]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


#X =  train_data[['Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Seats', 'Brand_Label']]
X = train_data[['Kilometers_Driven', 'Fuel_Type_CNG','Fuel_Type_LPG','Fuel_Type_Electric','Fuel_Type_Diesel','Fuel_Type_Petrol',
       'Transmission_Automatic','Transmission_Manual', 'Owner_Type_First','Owner_Type_Second','Owner_Type_Third','Owner_Type_Fourth & Above', 'Mileage', 'Engine', 'Seats']]
y =  pd.read_csv("training_data_targets.csv",header=None, names= ["price"])

model = LinearRegression()

param_grid = {
    'fit_intercept': [True, False]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

train_predictions = best_model.predict(X)

# Calculate Mean Squared Error (MSE) using the best model
mse = mean_squared_error(y, train_predictions)
print("Mean Squared Error (MSE) with Best Model:", mse)

rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y, train_predictions)
print(f"R-squared (R2) score on the test set: {r2:.4f}")

mae = mean_absolute_error(y, train_predictions)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# # Random Forest Regressor

# In[45]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


model = RandomForestRegressor(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],  
    'max_depth': [None, 10, 20],
    'max_features':['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X)

# Calculate evaluation metrics using the best model
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE) with Best Model:", mse)

rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y, y_pred)
print(f"R-squared (R2) score on the test set: {r2:.4f}")

mae = mean_absolute_error(y, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# # Decision Tree Regressor

# In[46]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)  

param_grid = {
    'max_depth': [2, 8, 9],  
    'min_samples_split': [5, 10, 15]  
    
}

grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

y_pred_dt = best_model.predict(X)

# Calculate evaluation metrics using the best model
mse = mean_squared_error(y, y_pred_dt)
print("Mean Squared Error (MSE) with Best Model:", mse)

rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y, y_pred_dt)
print(f"R-squared (R2) score on the test set: {r2:.4f}")

mae = mean_absolute_error(y, y_pred_dt)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# # K Nearest Neighbor

# In[47]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


knn_model = KNeighborsRegressor(n_neighbors=5)  

param_grid = {
    'n_neighbors': [7, 9, 11],  
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]  
}

grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

y_pred_km = best_model.predict(X)

# Calculate evaluation metrics using the best model
mse = mean_squared_error(y, y_pred_km)
print("Mean Squared Error (MSE) with Best Model:", mse)

rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y, y_pred_km)
print(f"R-squared (R2) score on the test set: {r2:.4f}")

mae = mean_absolute_error(y, y_pred_km)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# # Support Vector Regressor

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


svr_model = SVR()  

param_grid = {
    'C': [1, 10, 100],  
    'gamma': ['scale', 'auto'], 
    'kernel':['rbf','linear', 'poly']
}

grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

y_pred_svr = best_model.predict(X)

# Calculate evaluation metrics using the best model
mse = mean_squared_error(y, y_pred_svr)
print("Mean Squared Error (MSE) with Best Model:", mse)

rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y, y_pred_svr)
print(f"R-squared (R2) score on the test set: {r2:.4f}")

mae = mean_absolute_error(y, y_pred_svr)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# # RESULTS

# On comparing the Root Mean Square Error, Mean Absolute Error, Mean Square Error and R2 of all the above models we got the K-Nearest Neighbor model as a best model.

# In[ ]:


y_pred_km_test = best_model.predict(test_data2)


# In[ ]:


# Print the DataFrame line by line
for price in y_pred_km_test:
    print(price)


# In[ ]:


#Printing the predicted labels of the best model(Random Forest Regressor):-
np.savetxt("PredictedLabels.txt", y_pred_km_test, delimiter="\n", fmt="%.4f")

