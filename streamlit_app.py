#!/usr/bin/env python
# coding: utf-8

# ## Program Aplikasi Employee Attrittion

# In[1]:


#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,traceback
import seaborn as sns


# In[21]:


hr_data=pd.read_csv('train.csv')


# In[23]:


#Import Data
hr = hr_data
col_names = hr.columns.tolist()
print("Column names:")
print(col_names)

print("\nSample data:")
hr.head()


# In[24]:


hr=hr.rename(columns = {'Department':'Department'})
#Display data type for each column
hr.dtypes


# In[25]:


#Check for Missing Values
hr.isnull().any()


# In[26]:


#Dimensions of our dataset
hr.shape


# In[27]:


#Summary for each variable
hr.describe()


# In[28]:


#To get the unique values for department
hr['Department'].unique()


# In[29]:


#Combine "technical","support" and "IT" into one department
hr['Department']=np.where(hr['Department'] =='support', 'technical', hr['Department'])
hr['Department']=np.where(hr['Department'] =='IT', 'technical', hr['Department'])


# In[30]:


#Print the updated values of departments
print(hr['Department'].unique())


# # Data Exploration

# In[31]:


#Get a count the number of employees that stayed and left the company 0=No 1=Yes
hr['Attrition'].value_counts()


# In[32]:


hr.groupby('Attrition').mean()


# In[33]:


hr.groupby('Department').mean()


# In[34]:


hr.groupby('Salary').mean()


# ## Data Visualization

# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')

#Bar chart for department employee work for and the frequency of turnover
pd.crosstab(hr['Department'],hr['Attrition']).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')


# In[44]:


#Bar chart for employee salary level and the frequency of turnover
table=pd.crosstab(hr.Salary, hr.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


# In[43]:


hr.Salary


# In[57]:


#Proportion of employees left by department
pd.crosstab(hr.Department, hr.Attrition)


# In[58]:


#Histogram of numeric variables
num_bins = 10

hr.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()


# In[59]:


hr.head()


# In[60]:


cat_vars=['Department','Salary']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(hr[var], prefix=var)
    hr1=hr.join(cat_list)
    hr=hr1


# In[61]:


hr.drop(hr.columns[[7, 8]], axis=1, inplace=True)


# In[62]:


hr.columns.values


# In[63]:


hr.head()


# In[64]:


hr_vars=hr.columns.values.tolist()
y=['Attrition']
X=[i for i in hr_vars if i not in y]


# In[65]:


X


# ## Feature Selection

# In[66]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

rfe = RFE(model, 10)
rfe = rfe.fit(hr[X], hr[y])
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


cols=['satisfaction_level', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low','salary_medium'] 
X=hr[cols]
y=hr['Attrition']


# ### Logistic Regression Model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


#Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))


# ### Random Forest

# In[ ]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[ ]:


print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))


# ### Support Vector Machine

# In[ ]:


#SVM Classifier
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)


# In[ ]:


print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))


# ### XGBoost Classifier 

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb=XGBClassifier()
xgb.fit(X_train, y_train)


# In[ ]:


print('XGBoost accuracy: {:.3f}'.format(accuracy_score(y_test, xgb.predict(X_test))))


# ### 10 Fold Cross Validation

# In[ ]:


#For Random Forest
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("cross validation average accuracy for Random Forest Classifier: %.3f" % (results.mean()))


# In[ ]:


#For SVM
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = SVC()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("cross validation average accuracy for SVM Classifier: %.3f" % (results.mean()))


# In[ ]:


#For XGBoost
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = XGBClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("cross validation average accuracy for XGBoost Classifier: %.3f" % (results.mean()))


# ### Classification Report

# In[ ]:


#Classification report for Random Forest
from sklearn.metrics import classification_report
print(classification_report(y_test, rf.predict(X_test)))


# In[ ]:


#Confusion Matrix for Random Forest
y_pred = rf.predict(X_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')


# In[ ]:


#Classification report for Logistic Regression
print(classification_report(y_test, logreg.predict(X_test)))


# In[ ]:


#Confusion Matrix for Logistic Regression
logreg_y_pred = logreg.predict(X_test)
logreg_cm = metrics.confusion_matrix(logreg_y_pred, y_test, [1,0])
sns.heatmap(logreg_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('logistic_regression')


# In[ ]:


#Classification report for SVM
print(classification_report(y_test, svc.predict(X_test)))


# In[ ]:


#Confusion Matrix for SVM
svc_y_pred = svc.predict(X_test)
svc_cm = metrics.confusion_matrix(svc_y_pred, y_test, [1,0])
sns.heatmap(svc_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.savefig('support_vector_machine')


# In[ ]:


print(classification_report(y_test, xgb.predict(X_test)))


# In[ ]:


#Confusion Matrix for XGBoost Classifier
xgb_y_pred = xgb.predict(X_test)
xgb_cm = metrics.confusion_matrix(xgb_y_pred, y_test, [1,0])
sns.heatmap(xgb_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('XGBoost Classifier')
plt.savefig('XGBoost_Classifier')


# ### Variable Importance for Random Forest Classifier

# In[ ]:


feature_labels = np.array(['satisfaction_level', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low','salary_medium'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))


# ### Variable Importance for XGBoost Classifier

# In[ ]:


feature_labels = np.array(['satisfaction_level', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low','salary_medium'])
importance = xgb.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))


# In[ ]:





# ## Hperparameter Tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


#Randomized Search CV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf=RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='accuracy', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


rf.fit(X,y)


# In[ ]:


rf.best_score_


# In[ ]:


rf.best_params_

