#!/usr/bin/env python
# coding: utf-8

# # Bharat intern
# ## Task 2 - Titanic classification

# In[67]:


# importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


# loading datasets
titanic = pd.read_csv(r"C:\Users\HP\Downloads\Titanic-Dataset.csv")


# In[69]:


titanic.head()


# In[70]:


titanic.shape


# In[71]:


# statistical info
titanic.describe()


# In[72]:


# datatype information
data.info()


# In[73]:


titanic.columns


# ### Exploratory Data Analysis

# In[74]:


titanic['Survived'].value_counts()


# In[75]:


titanic.groupby('Survived').mean()


# In[76]:


num_male = len(titanic[titanic['Sex']=='male'])
print("number of males in titanic:",num_male)


# In[77]:


num_female= len(titanic[titanic['Sex']=='female'])
print("number of females in titanic:",num_female)


# In[78]:


#Plotting
plt.hist(titanic.Sex)


# In[79]:


survived = len(titanic[titanic['Survived']==1])
non_survived =len(titanic[titanic['Survived']==0])
titanic.groupby('Sex')[['Survived']].mean()


# In[80]:


# count plot of survived and not survived
sns.countplot(x ='Survived', data= titanic)


# In[81]:


# male vs Female survival
sns.countplot(x='Survived',data= titanic, hue='Sex')


# In[82]:


plt.figure(1)
age  = titanic.loc[titanic.Survived == 1, 'Age']
plt.title('The histogram of the age groups of the people that had survived')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))


plt.figure(2)
age  = titanic.loc[titanic.Survived == 0, 'Age']
plt.title('The histogram of the age groups of the people that coudn\'t survive')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))


# In[83]:


titanic[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[84]:


titanic[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[85]:


titanic[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[86]:


sns.countplot(x='Survived',hue = 'Pclass', data = titanic)


# ### Data preprocessing

# In[87]:


# checking missing values
titanic.isna().sum()


# ### visulizing null values

# In[88]:


# visulizing using heatmap
sns.heatmap(titanic.isna())


# In[89]:


# finding the %  of null values
(titanic['Age'].isna().sum()/len(titanic['Age']))*100


# In[90]:


# finging the % of null values in Cabin
(titanic['Cabin'].isna().sum()/len(titanic['Cabin']))*100


# here we can see that 77% of missing values in cabin column, actually the column is not necessory for our prediction. There for dropping unwanted columns

# In[92]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace = True)
titanic['Age'].isna().sum()


# ### converting sex column into numerical values 

# In[94]:


gender = pd.get_dummies(titanic['Sex'],drop_first = True)
titanic['Gender'] = gender


# ### Droping the columns which are not  required

# In[95]:


titanic.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis = 1, inplace = True)


# In[96]:


titanic.head()


# ### Seperating Dependent and Independent variables

# In[98]:


x = titanic[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y = titanic['Survived']


# ### Data modelling

# In[99]:


# Training testing and spliting the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)


# In[100]:


# using logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy score:", accuracy_score(y_test, y_pred))


# In[104]:


# confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
print(classification_report(y_test, y_pred))


# In[106]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train,y_train)

pred_y = model1.predict(x_test)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(y_test,pred_y))


# In[107]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,pred_y)
print(confusion_mat)
print(classification_report(y_test,pred_y))


# In[109]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred3))


# In[110]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred3)
print(confusion_mat)
print(classification_report(y_test,y_pred3))


# In[111]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=42)
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred4))


# In[112]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred4)
print(confusion_mat)
print(classification_report(y_test,y_pred4))


# In[ ]:




