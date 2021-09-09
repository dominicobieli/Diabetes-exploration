#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries for the Data Exploration 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

import numpy as np
data = pd.read_csv('diabetes.csv')


# In[2]:


data.shape


# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


data.describe()


# # Dataset Interrogation 

# In[6]:


print(data.isnull().sum())


# In[7]:


data.isnull().values.any()


# In[8]:


print('Number of zero values for glucose: ' + str(len(data[data['Glucose']==0])))
print('Number of zero values for BloodPressure: ' + str(len(data[data['BloodPressure']==0])))
print('Number of zero values for SkinThickness: ' + str(len(data[data['SkinThickness']==0])))
print('Number of zero values for Insulin: ' + str(len(data[data['Insulin']==0])))
print('Number of zero values for BMI: ' + str(len(data[data['BMI']==0])))


# # Filling the Zero value with the mean - Not applied to the overall Dataset! 

# In[10]:


nonzero_mean = data[ data.Insulin != 0 ].mean()
data.loc[ data.Insulin == 0, "Insulin" ] = nonzero_mean

nonzero_mean = data[ data.Glucose != 0 ].mean()
data.loc[ data.Glucose == 0, "Glucose" ] = nonzero_mean

nonzero_mean = data[ data.BloodPressure != 0 ].mean()
data.loc[ data.BloodPressure == 0, "BloodPressure" ] = nonzero_mean

nonzero_mean = data[ data.SkinThickness!= 0 ].mean()
data.loc[ data.SkinThickness == 0, "SkinThickness" ] = nonzero_mean

nonzero_mean = data[ data.BMI != 0 ].mean()
data.loc[ data.BMI == 0, "BMI" ] = nonzero_mean


# In[11]:


print('Number of zero values for glucose: ' + str(len(data[data['Glucose']==0])))
print('Number of zero values for BloodPressure: ' + str(len(data[data['BloodPressure']==0])))
print('Number of zero values for SkinThickness: ' + str(len(data[data['SkinThickness']==0])))
print('Number of zero values for Insulin: ' + str(len(data[data['Insulin']==0])))
print('Number of zero values for BMI: ' + str(len(data[data['BMI']==0])))


# # Statistical Analysis

# In[9]:


hist = data.hist(figsize=(10,8))


# In[10]:


data.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# In[11]:


plt.figure(figsize=(10,6))
sns.distplot(data['BMI'],kde=False,bins=50)
plt.title('BMI of Pima People')
plt.ylabel('Number of People')
plt.show()
print('Average BMI: ' + str(data['BMI'].mean()))


# In[12]:


plt.figure(figsize=(8,6))
sns.countplot(x='Outcome',data=data)
plt.title('Healthy or Diabetic')
plt.ylabel('Number of People')
plt.show()
print('Ratio of Diabetic Population: ' + str(len(data[data['Outcome']==1])/len(data)))


# In[13]:


x = sns.FacetGrid(data, col="Outcome",size=5)
x = x.map(plt.hist, "Glucose",bins=25)


# In[14]:


x = sns.FacetGrid(data, col="Outcome",size=5)
x = x.map(plt.hist, "BMI",bins=25)


# In[16]:


Corr=data[data.columns].corr() 
sns.heatmap(Corr, annot=True)


# # Data Training 

# In[17]:


# Split
attributes = list(data.columns[:8])
X = data[attributes].values 
y= data['Outcome'].values


# In[18]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X = sc_X.fit_transform(X) 


# In[19]:


# Split into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


# In[20]:


# Import algorithms.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# In[29]:


# Create models in question.
models = []
models.append(("Nab",GaussianNB()))
models.append(("SVM Linear",SVC(kernel="linear")))
models.append(("SVM RBF",SVC(kernel="rbf")))
models.append(("Random Forest",  RandomForestClassifier()))


# # Evaluation of Classification

# In[32]:


# Find accuracy of models.
results = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    results.append(tuple([name,cv_result.mean(), cv_result.std()]))
  
results.sort(key=lambda x: x[1], reverse = True)    
for i in range(len(results)):
    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))


# In[31]:


from sklearn.model_selection import GridSearchCV
model = SVC()
paramaters = [
             {'C' : [0.01, 0.1, 1, 10, 100, 1000], 'kernel' : ['linear']}                                       
             ]
grid_search = GridSearchCV(estimator = model, 
                           param_grid = paramaters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_  

print('Best accuracy : ', grid_search.best_score_)
print('Best parameters :', grid_search.best_params_  )


# In[24]:


# Predict  
final_model = SVC(C = 0.1, kernel = 'linear')
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
print(accuracy_score(y_test, y_pred) * 100) 


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)


# In[25]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics # for the check the error and accuracy of the model


# In[26]:


Ran = RandomForestClassifier(n_estimators=50)
Ran.fit(X_train, y_train)
y_pred = Ran.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation 
cv_scores =cross_val_score(Ran, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)), 
      ", Standard deviation: {}".format(np.std(cv_scores)))

ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="coolwarm", fmt="d", 
            xticklabels = ['0', '1'], yticklabels = ['0', '1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix");


# In[27]:


SVM = SVC(C = 0.1, kernel = 'linear')
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation 
cv_scores =cross_val_score(SVM, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)), 
      ", Standard deviation: {}".format(np.std(cv_scores)))

ConfMatrix = confusion_matrix(y_test,SVM.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="coolwarm", fmt="d", 
            xticklabels = ['0', '1'], yticklabels = ['0', '1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix");


# In[28]:


Nab = GaussianNB()
Nab.fit(X_train, y_train)
y_pred = Nab.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation 
cv_scores =cross_val_score(Nab, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)), 
      ", Standard deviation: {}".format(np.std(cv_scores)))

ConfMatrix = confusion_matrix(y_test,Nab.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="coolwarm", fmt="d", 
            xticklabels = ['0', '1'], yticklabels = ['0', '1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix");

