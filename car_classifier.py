/# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:28:28 2020

@author: mk626
"""

# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

import os
os.chdir('PycharmProjects\CarValue')


#setting size of all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data=pd.read_csv('cars_sampled.csv',na_values=['??','???'])

cars_data2=cars_data.copy(deep=True)
cars_data2.head()

cars_data2.info()

cars_data2.describe()

#to get rid off scientific notation 
pd.set_option('display.float_format',lambda x :'%.3f' % x)  

#to set maximum number of columns display same time on the screen 
pd.set_option('display.max_columns',500)
                     
pd.set_option('display.max_rows',500)

#drop unnecessary columns 
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars_data2.drop(cols,axis=1)


#to remove duplicate record 470 duplicate records are there
cars.drop_duplicates(keep='first',inplace=True)

cars.isnull().sum()

#year_wise count means how many cars has been sold in year
year_wise=cars['yearOfRegistration'].value_counts().sort_index()
year_wise

before_1950=sum(cars['yearOfRegistration']<1950)
after_2018=sum(cars['yearOfRegistration']>2018)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=(False),data=cars)

#price_count
# 
price_count=cars['price'].value_counts().sort_index()

pd.set_option('display.max_rows',500)
sns.distplot(cars['price'])
cars['price'].describe()
# cars.describe()
#boxplot
sns.boxplot(y=cars['price'])
expensive=sum(cars['price']>150000)
cheap=sum(cars['price']<100)




#powerPS count
power_count=cars['powerPS'].value_counts().sort_index()

sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=True,data=cars)
less_than=sum(cars['powerPS']<10)

greater_than=sum(cars['powerPS']>500)



cars=cars[(cars.yearOfRegistration<=2018)&(cars.yearOfRegistration>=1950)&(cars.price>=100)&(cars.price<=150000)&(cars.powerPS>=10)&(cars.powerPS<=500)]

cars['monthOfRegistration']/=12

cars['age']=(2018-cars['yearOfRegistration']+cars['monthOfRegistration'])

cars['age']=round(cars['age'],2)
cars['age'].describe()

#dropping columns year and month of registration 
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

cars.isnull().sum()

#visualization of data

#age
sns.distplot(cars['age'])
sns.boxplot(y=cars['age'])


#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#age vs price
sns.regplot(x='age',y='price',scatter=True,fit_reg=False,data=cars)
#sns.regplot(x='age',y='price',scatter=True,fit_reg=False,marker='*',data=cars)

#powerPS vs Price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)


#variable seller
cars['seller'].value_counts()

pd.crosstab(cars['seller'],columns='counts',normalize=True)
sns.countplot(x=cars['seller'],data=cars) #this variable is insignificant as all the cars are pvt


#offer variable
cars['offerType'].value_counts()
sns.countplot(x=cars['offerType'],data=cars)
#this variable is also insignificant 

#abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='counts',normalize=True)
sns.countplot(x=cars['abtest'],data=cars)
#equally distributed 
sns.boxplot(x='abtest',y='price',data=cars)
sns.regplot(x='abtest',y='price',scatter=True,fit_reg=False,data=cars)
#for every price value there is 50-50 distribution hence it is also insignificant

#vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='counts',normalize=True)
sns.countplot(x=cars['vehicleType'],data=cars)
# sns.countplot(x=cars['vehicleType'],hue='price',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)


#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
# sns.regplot(x='gearbox',y='price',scatter=True,fit_reg=False,data=cars)
#gearbox affect the price of the cars


#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns='counts',normalize=True)
sns.countplot(y=cars['model'],data=cars)
sns.boxplot(x='model',y='price',data=cars)



#variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',data=cars,fit_reg=False,scatter=True)

#kilometer affect the price of the cars so considered

#fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns='counts',normalize=True)
sns.countplot(x=cars['fuelType'],data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
sns.regplot(x='fuelType',y='price',data=cars,fit_reg=False,scatter=True)

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='counts',normalize=True)
sns.countplot(y=cars['brand'],data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#consider it foe modelling 


#car damages or not 
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='counts',normalize=True)
sns.countplot(x=cars['notRepairedDamage'],data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)


#remove variable 
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)

cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
# cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)

cars_select1.corr().loc[:,'powerPS'].abs().sort_values(ascending=False)[1:]


cars.isnull().sum()

cars['vehicleType'].mode()

#beginning of the model building
# by dropp ing missing values
# by filling with mean, median and mode values 






cars_omits=cars.dropna(axis=0)

# =========================================================
# converting categrial value into int or float()
# =========================================================
from sklearn import preprocessing 


category_col=cars_omits.select_dtypes(exclude=['int64','float64'])

category_col_name=list(category_col.columns)

labelEncoder = preprocessing.LabelEncoder() 


mapping_dict={}

for col in category_col_name:
    cars_omits[col]=labelEncoder.fit_transform(cars_omits[col])
    
    le_name_mapping = dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_))) 
    mapping_dict[col]= le_name_mapping
    
print(mapping_dict)


#converting categorial variable into integer variable 
# cars_omits=pd.get_dummies(cars_omits,drop_first=True)


#==================================================================
# start from here==============================================
#================================================================





#colunms_list=list(cars_omits.columns)
#features=list(set(colunms_list)-set(['price']))
#x1=cars_omits[features].values()
#y1=cars_omits['price'].values()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
 
x1=cars_omits.drop(['price'],axis='columns',inplace=False)


y1=cars_omits['price']

prices=pd.DataFrame({"1. before":y1,"2. After ":np.log(y1)})

prices.hist()

y1=np.log(y1)


X_train,X_test,y_train,y_test=train_test_split(x1, y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#baseline model for omitted data

base_predict=np.mean(y_test)

base_predict

base_predict=np.repeat(base_predict,len(y_test))
# finding rmse value

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test,base_predict))
print(base_root_mean_square_error)

#base root mean squared Error is 1.27448.... this is our comparing parameter  



lgr=LinearRegression(fit_intercept=(True))

#model
model_lin1=lgr.fit(X_train,y_train)

cars_prediction=lgr.predict(X_test)


#now computing rmse value
lin_mse1=mean_squared_error(y_test, cars_prediction)

lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)


#calculating r Squared value for our model to know how good our model is working 

r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print("model on test set=",r2_lin_test1,"\nmodel on train set=",r2_lin_train1)

# computing residual 
residual=y_test-cars_prediction
sns.regplot(x=cars_prediction,y=residual,scatter=(True),fit_reg=(False),data=cars)

#dumping linear model for prediction 
#=========================================================
pickle.dump(lgr, open('ln_wl_model.pkl','wb'))
#=========================================================


#==================================================================
# Random Forest model
#==============================================
rf=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=(100),min_samples_split=10,
                         min_samples_leaf=4,random_state=(1))

model_rf=rf.fit(X_train,y_train)

cars_prediction_rf1=rf.predict(X_test)

rf_mse1=mean_squared_error(y_test,cars_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#============dumping rf model for prediction========================================
pickle.dump(rf, open('rfmodel.pkl','wb'))
#===================================================
#computing r2 value for random forest model

r2_rf_test1=model_rf.score(X_test,y_test)
r2_rf_train1=model_rf.score(X_train,y_train)
print("model on test set=",r2_rf_test1,"\nmodel on train set=",r2_rf_train1)


#==================================================================
# For imputed values
#==================================================================














cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float64' else x.fillna(x.value_counts().index[0]))




category_col=cars_imputed.select_dtypes(exclude=['int64','float64'])

category_col_name=list(category_col.columns)

labelEncoder = preprocessing.LabelEncoder() 


mapping_dict={}

for col in category_col_name:
    cars_imputed[col]=labelEncoder.fit_transform(cars_imputed[col])
    
    le_name_mapping = dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_))) 
    mapping_dict[col]= le_name_mapping
    
print(mapping_dict)











# cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)
# cars['fuelType'].value_counts().index[0]

#==================================================================
# Linear regression  model
#==============================================

x2=cars_imputed.drop(['price'],'columns',inplace=False)
y2=cars_imputed['price']

prices=pd.DataFrame({"1. before":y2,"2. After ":np.log(y2)})

prices.hist()

# y2=np.log(y2)



#splitting data into train and test data set
X_train1,x_test1, y_train1,y_test1= train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape,x_test1.shape, y_train1.shape,y_test1.shape)


#now making base line model to compare 

base_predict2=np.mean(y_test1)
base_predict2


base_predict2=np.repeat(base_predict2,len(y_test1))
# finding rmse value

base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1,base_predict2))
print(base_root_mean_square_error_imputed)


lgr2=LinearRegression(fit_intercept=(True))

model_lin2=lgr2.fit(X_train1,y_train1)


cars_prediction_lin2=lgr2.predict(x_test1)


root_mean_square_error_imputed_lin2 = np.sqrt(mean_squared_error(y_test1,cars_prediction_lin2))


r2_l2_test1=model_lin2.score(x_test1,y_test1)
r2_l2_train1=model_lin2.score(X_train1,y_train1)
print("model on test set=",r2_l2_test1,"\nmodel on train set=",r2_l2_train1)

#============dumping linear 2(imputed) model  for prediction========================================
pickle.dump(lgr2, open('ln2_wl_model.pkl','wb'))
#===================================================

# ================================================
# random Forest on imputed data
# =================================================

rf2=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=(100),min_samples_split=10,
                         min_samples_leaf=4,random_state=(1))

model_rf2=rf2.fit(X_train1, y_train1)

cars_prediction_rf2=rf2.predict(x_test1)





root_mean_square_error_imputed_rf2 = np.sqrt(mean_squared_error(y_test1,cars_prediction_rf2))

print(root_mean_square_error_imputed_rf2)



r2_rf2_test1=model_rf2.score(x_test1,y_test1)
r2_rf2_train1=model_rf2.score(X_train1,y_train1)
print("model on test set=",r2_rf2_test1,"\nmodel on train set=",r2_rf2_train1)















