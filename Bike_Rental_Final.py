
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime

# To read data from excel file
bike_data = pd.read_csv('D:\#\Projects\Projects\Bike Prediction\Python\day.csv')
bike_data.head()
bike_data.describe()
bike_data.shape[0]
bike_data.shape[1]
bike_data.isnull().sum()

#in dataset we have information dor some colum that containts value which are nesessary to convert
#atemp: Normalized feeling temperature in Celsius.
#The values are derived via(t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
bike_data.atemp = bike_data.atemp*(50-(-16))+(-16)
#hum: Normalized humidity. The values are divided to 100
bike_data.hum = bike_data.hum*100
#(max)windspeed: Normalized wind speed. The values are divided to 67
bike_data.windspeed = bike_data.windspeed*67
#temp: Normalized temperature in Celsius
#The values are derived via(t-t_min)/(t_max-t_min),t_min=-8, t_max=+39 (only in hourly scale)
bike_data.temp = bike_data.temp *(39-(-8))+(-8)

bike_data.head()

#data Binning now we convert categorical variable to numric variable 
bike_data.replace({'holiday' : 1}, 'holiday', inplace= True)
bike_data.replace({'holiday' : 0}, 'commanday', inplace=True)
# This shows any given day is either 'working' or 'week off' or 'holiday'.
print('Holidays'+format(bike_data[bike_data.holiday == 'holiday'].weekday.unique()))
print('CommanDays'+format(bike_data[bike_data.holiday == 'commanday'].weekday.unique()))
# 0 It can be a week off or a holidy.
# 1 is neither weekend nor holiday. Hence it is working day
bike_data.replace({'workingday': 0}, 'no working day', inplace = True)
bike_data.replace({'workingday': 1}, 'working day', inplace = True)



# Converting weather situation
bike_data.replace({'weathersit': 1}, 'Pleasant', inplace = True)
bike_data.replace({'weathersit': 2}, 'Moderate', inplace = True)
bike_data.replace({'weathersit': 3}, 'Bad', inplace = True)
bike_data.replace({'weathersit': 4}, 'Extreme', inplace = True)

# Converting yr
bike_data.replace({'yr' : 0}, '2011', inplace= True)
bike_data.replace({'yr' : 1}, '2012', inplace= True) 

# As we have two different temperature with a slight difference, I will be using a new temperature variable
# This variable will be derived by calculating the mean of the 'temperature' and 'feeling temperature'
bike_data['avragetemp'] = (bike_data.temp + bike_data.atemp)/2
bike_data = bike_data[['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit','avragetemp', 'temp', 'atemp','hum', 'windspeed',
       'casual', 'registered', 'cnt']]

bike_data.head()

# Converting month
months = ['January', 'February', 'March','April','May','June','July','August','September','October','November','December'] 
for i in range(len(months)): 
    bike_data.replace({'mnth': i+1} ,months[i], inplace = True) 


# Converting Weekday  
weeks = ['Sunday','Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday']
for i in range(len(weeks)):
    bike_data.replace({'weekday': i}, weeks[i], inplace = True)

# Converting the variable season
seasons = ['Spring','Summer','Fall', 'Winter']
for i in range(len(seasons)):
    bike_data.replace({'season': i+1} ,seasons[i], inplace = True)

bike_data.head()

#visualization
#normal distrubiton curve
b,abc = plt.subplots(figsize= (30,7))
sns.distplot(bike_data.cnt, bins= 75)


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(bike_data['cnt'], bike_data['temp'], ax=ax)
sns.rugplot(bike_data['cnt'], color="g", ax=ax)
sns.rugplot(bike_data['temp'], vertical=True, ax=ax)


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(bike_data['temp'], bike_data['cnt'], cmap=cmap, n_levels=60, shade=True);



g = sns.PairGrid(bike_data, vars=['temp','atemp','avragetemp','windspeed','registered','casual','hum'],hue='cnt', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend()


g = sns.PairGrid(bike_data)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(bike_data['temp'], bike_data['cnt'], ax=ax)
sns.rugplot(bike_data['temp'], color="g", ax=ax)
sns.rugplot(bike_data['cnt'], vertical=True, ax=ax)


fig = plt.figure(figsize=(20, 5))
sns.catplot(x="season", y="cnt", kind="swarm", data=bike_data)
sns.catplot(x="season", y="cnt", kind="boxen",data=bike_data.sort_values("cnt"))
sns.catplot(x="yr", y="cnt",hue="yr", kind="swarm", data=bike_data)
sns.catplot(x="dteday", y="cnt", kind="swarm", data=bike_data)
sns.catplot(x="workingday", y="cnt",hue="workingday", kind="swarm", data=bike_data)
sns.catplot(x="mnth", y="cnt",hue="workingday", kind="swarm", data=bike_data)
sns.catplot(x="weekday", y="cnt",hue="workingday", kind="swarm", data=bike_data)
sns.catplot(x="weathersit", y="cnt",hue="workingday", kind="swarm", data=bike_data)




#Outlier Analysis

boxplot = bike_data.boxplot(column=['cnt'])
boxplot = bike_data.boxplot(column=['hum'])
boxplot = bike_data.boxplot(column=['windspeed'])
boxplot = bike_data.boxplot(column=['temp'])
boxplot = bike_data.boxplot(column=['atemp'])
boxplot = bike_data.boxplot(column=['avragetemp'])
boxplot = bike_data.boxplot(column=['casual'])
boxplot = bike_data.boxplot(column=['registered'])

bike_data.columns

#CORRELATION
corre = ['temp', 'atemp','avragetemp', 'hum', 'windspeed', 'casual', 'registered','cnt']

data_corre = bike_data.loc[:,corre]

f, ax = plt.subplots(figsize=(18,6))
corr = data_corre.corr().abs()


# Select upper triangle of correlation matrix
lower = corr.where(np.tril(np.ones(corr.shape), k=0).astype(np.bool))


df = bike_data
df.avragetemp = (df.avragetemp-(df.avragetemp.min()))/((df.avragetemp.max())-(df.avragetemp.min()))
df.dteday = df.dteday.astype('object')


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = df[['dteday', 'season', 'yr','mnth', 'holiday', 'weekday','workingday', 'weathersit','avragetemp', 'hum','windspeed','cnt']]
df_object = df
df_object_target = bike_data['cnt']
df_object = bike_data.drop(['cnt'], axis=1)
cate_columns = [col for col in df.columns.values if df[col].dtype == 'object']
df_object_cat = df[cate_columns]
df_object_num = df.drop(cate_columns, axis=1)
df_object_num = df_object_num.drop('cnt', axis = 1)
df_object_cat_dummies = pd.get_dummies(df_object_cat,drop_first=True)
new_bike_data = pd.concat([df_object_num, df_object_cat_dummies], axis=1)

#split the dataset into train & test
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(new_bike_data, df_object_target, test_size = 0.30, random_state=0)


#Linear Regression model
import statsmodels.api as sm
model1 = sm.OLS(y_train, x_train).fit()
model1.summary()
#
def rmse(predictions, targets):
    differences = predictions - targets # the DIFFERENCEs.
    differences_squared = differences ** 2 # the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean() # the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared) # ROOT of ^
    return rmse_val

cols = ['Model', 'R-Squared Value', 'Adj.R-Squared Value', 'RMSE']
models_report = pd.DataFrame(columns = cols)
predictions1 = model1.predict(x_test)
tmp1 = pd.Series({'Model': " Base Linear Regression Model",'R-Squared Value' : model1.rsquared,'Adj.R-Squared Value': model1.rsquared_adj,'RMSE': rmse(predictions1, y_test)})
model1_report = models_report.append(tmp1, ignore_index = True)
model1_report

# Adding connstant
df_object_constant = sm.add_constant(new_bike_data)

x_train1,x_test1, y_train1, y_test1 = train_test_split(df_object_constant, df_object_target, test_size = 0.30, random_state=0)

# Linear Regression model 
import statsmodels.api as sm
model2 = sm.OLS(y_train1, x_train1).fit()
model2.summary2()


# Predicting the model on test data
predictions2 = model2.predict(x_test1)
info_title = pd.Series({'Model': " Linear Regression Model with Constant",'R-Squared Value' : model2.rsquared,'Adj.R-Squared Value': model2.rsquared_adj,'RMSE': rmse(predictions2, y_test1)})
model2_report = models_report.append(info_title, ignore_index = True)
model2_report

new_bike_data = new_bike_data[['avragetemp', 'hum', 'windspeed','dteday_2011-01-06','dteday_2011-10-11', 'dteday_2012-01-17','dteday_2012-12-31', 'season_Spring','yr_2012', 'mnth_July','weekday_Saturday','workingday_working day', 'weathersit_Moderate', 'weathersit_Pleasant']]

#Let us now split the dataset into train & test
from sklearn.cross_validation import train_test_split
a_train,a_test, b_train, b_test = train_test_split(new_bike_data, df_object_target, test_size = 0.30, random_state=0)

import statsmodels.api as sm
model3 = sm.OLS(b_train, a_train).fit()
model3.summary()

predictions3 = model3.predict(a_test)
info_title = pd.Series({'Model': " Linear Regression Model with selected features",'R-Squared Value' : model3.rsquared,'Adj.R-Squared Value': model3.rsquared_adj,'RMSE': rmse(predictions3, b_test)})
model3_report = models_report.append(info_title, ignore_index = True)
model3_report


new_bike_data = new_bike_data[['avragetemp', 'windspeed','dteday_2011-01-06', 'season_Spring','yr_2012', 'mnth_July','weekday_Saturday','workingday_working day', 'weathersit_Moderate', 'weathersit_Pleasant']]

#Let us now split the dataset into train & test
from sklearn.cross_validation import train_test_split
c_train,c_test, d_train, d_test = train_test_split(new_bike_data, df_object_target, test_size = 0.30, random_state=0)


model4 = sm.OLS(d_train, c_train).fit()
model4.summary()


predictions4 = model4.predict(c_test)
tmp2 = pd.Series({'Model': " Final Optimized Linear Regression Model with reduced features",'R-Squared Value' : model4.rsquared,'Adj.R-Squared Value': model4.rsquared_adj,'RMSE': rmse(predictions4, d_test)})
model4_report = models_report.append(tmp2, ignore_index = True)
model4_report

def MAPE(y_true,y_pred):
    mape = np.mean(np.abs(y_true-y_pred)/y_true)
    return mape

accuracy = 1-MAPE(d_test,predictions4)
print(format(round(accuracy*100 ,2))+' %')
residuals  = predictions4-d_test
fig = plt.figure()
axes = fig.add_axes([0, 0, 1.0, 1.0]) 
plt.scatter(residuals,d_test)
axes.axhline(color="blue", ls="--")
res = model4.resid

import scipy.stats as stats
fig = sm.qqplot(res, stats.t, fit=True, line='45')
plt.title('Quantiles Vs Residuals')
plt.xlabel('Quantiles')
plt.ylabel('Residuals');
plt.show()
X = df.iloc[:,:-1].values #independent variable
y = df.iloc[:,11].values     #dependents variable
le = LabelEncoder()
c_cat = [0,1,2,3,4,5,6,7]
for i in c_cat:
    X[:,i] = le.fit_transform(X[:,i]).astype('str')

oh_en = OneHotEncoder()
X = oh_en.fit_transform(X).toarray()

# Avoiding the dummy variale 
X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 300, random_state=0)
regressor.fit(X_train,y_train)
pr_cnt = regressor.predict(X_test)
accuracy = 1-MAPE(y_test,pr_cnt)
print(format(round(accuracy*100 ,2))+' %')
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= regressor, X= X_train, y = y_train, cv = 7)
# accuracy of random Forest regression 
print(abs(accuracies).mean()*100)
print(abs(accuracies).std()*100)