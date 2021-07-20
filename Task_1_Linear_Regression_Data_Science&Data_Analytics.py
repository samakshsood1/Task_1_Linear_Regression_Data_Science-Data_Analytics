#!/usr/bin/env python
# coding: utf-8

# # GRIP - THE SPARK FOUNDATION
# 
# ## DATA SCIENCE AND BUISNESS ANALYTICS INTERNSHIP

# ## **TASK 1 - Prediction using Supervised ML**
# 
# To Predict the percentage of marks of the students based on the number of hours they studied
# ## By - SAMAKSH SOOD
# 

# ### In this regression task we will predict the percentage of marks that a student is expected to score based on the number of hours the studied. this is a simple linear regression task as it involves just two variables. 

# ## Import the required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading The Data from data source

# In[2]:


# Reading the Data 
student = pd.read_csv ('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
student.head(10)


# In[3]:


student.describe()


# In[4]:


student.shape


# In[5]:


student.info()


# ## Check if there any null value in the Dataset

# In[6]:


student.isnull().sum()


# # Data Visualization

# In[7]:


plt.style.use('ggplot')
student.plot(kind='line')
plt.title('Study Hours vs Score')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# **Lets plot a regression line to see the correlation.**

# In[8]:


sns.regplot(x= student['Hours'], y= student['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Percentage Score', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(student.corr())


# **It is confirmed that the variables are positively correlated.**

# ## Data Visualisation with area plot

# In[9]:


xmin = min(student.Hours)
xmax = max(student.Hours)
student.plot(kind='area',alpha=0.8,stacked=True,figsize=(10,5),xlim=(xmin,xmax))
plt.title('Hours vs Score',size=15)
plt.xlabel('Hours',size=15)
plt.ylabel('Score',size=15)
plt.show()


#  By Visualization we come to know that this problem can be easily solved by linear regression

# ## Modeling the data

# ## Training the Model
# ### 1) Splitting the Data

# In[10]:


# Defining X and y from the Data
x = student.iloc[:, :-1].values  
y = student.iloc[:, 1].values

# Spliting the Data in two
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=2)


# ### 2) Fitting the Data

# In[11]:


regression = LinearRegression()
regression.fit(train_x, train_y)
print("---------Model Trained---------")
print('coehhicient: ', regression.coef_)
print('Intercept: ',regression.intercept_)


# ## we can also plot the fit line over the data in single linear regression 

# In[12]:


student.plot(kind='scatter',x='Hours',y='Scores',figsize=(5,4),color='r')
plt.plot(train_x, regression.coef_[0]*train_x + regression.intercept_,color='b')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# Blue line is the best fit line for the data

# ## Predicting the Percentage of Marks

# In[13]:


pred_y = regression.predict(test_x)
prediction = pd.DataFrame({'Hours': [i[0] for i in test_x], 'Predicted Marks': [k for k in pred_y]})
prediction


# ## Comparing the Predicted Marks with the Actual Marks

# In[14]:


compare_scores = pd.DataFrame({'Actual Marks': test_y, 'Predicted Marks': pred_y})
compare_scores


# ## Visually Comparing the Predicted Marks with the Actual Marks

# In[15]:


plt.scatter(x=test_x, y=test_y, color='blue')
plt.plot(test_x, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# ## Evaluation of model

# In[16]:


# Using metrics to find mean obsolute error and r2 to see the accuracy

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(test_y,pred_y))
print("R2-score: %.2f" % r2_score(pred_y, test_y))


# **Small value of Mean absolute error states that the chances of error or wrong forecasting through the model are very less. Higher the r2 value higher is the accuracy of model**

# ## What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[17]:



hours = 9.25
predicted_score = regression.predict([[hours]])
print(f'No. of hours = {hours}')
print(f'predicted Score  = {predicted_score[0]}')


# **According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.**
