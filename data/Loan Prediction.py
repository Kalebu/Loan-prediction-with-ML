
# coding: utf-8

# In[1]:


#===========Importing Neccessaryy library and module========

#===========Importing Classifier=========
from sklearn.ensemble import RandomForestClassifier

#==========Importing numpy for multidimensional array
import numpy as np

#=========importing matplotlib for Plotting Graph==========
import matplotlib.pyplot as plt

#=========Loadig model selection for splitting the dataset==============
from sklearn.model_selection import train_test_split

#=========Importing models for classification efficiency
from sklearn.metrics import accuracy_score, classification_report

#========Importing library for dealing with DataFrame=====
import pandas as pd


# In[2]:


#========Loading the dataset======
loan = pd.read_csv('loan.csv')


# In[3]:


#==========Visualising the Dataset======
loan.head(5)


# In[4]:


len(loan)


# In[132]:


"""
We have to check if the dataset contain null values so as we 
can replace them with mean values to improve our classification 

and 

we have to Encode string with Numerical values such as Gender and Education
and Loan status
"""


# In[5]:


#=======Checking the presence of Null Values=========

#==== We use built in Function isnull() and sum()

loan.isnull().sum()


# In[6]:


#========filling null values in Column with Numerical Values============

#===========Loading the Required libray

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
loan_n = loan.iloc[:, 8:11]
dependat =loan['Dependents']
dependat = dependat.replace('3+', 3)
loan_n1 = loan.iloc[:, 3:4]
imputer.fit(loan_n)
loan_n = imputer.transform(loan_n)
loan_d = pd.DataFrame(loan_n)


# In[7]:


loan.columns


# In[8]:


loan['LoanAmount'],loan['Loan_Amount_Term'], loan['Credit_History'] = loan_d[0], loan_d[1], loan_d[2]


# In[9]:


loan.isnull().sum()


# In[10]:


employ = loan['Self_Employed']
P_area = loan['Property_Area']
gender = loan["Gender"]
status = loan["Married"]
education = loan["Education"]
loan_status = loan['Loan_Status']


status = status.fillna("Yes")
gender = gender.fillna("Male")
P_area= P_area.fillna("Rural")
employ= employ.fillna("Yes")
dependat = dependat.fillna(1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

#========Initializing String to be Encoded==========

input_class = ["Male", "Female"]
input_class1 = ["Yes", "No"]
input_class2 = ["Rural", "Urban", "Semiurban"]
input_class3 = ["Graduate", "Not Graduate"]
input_class4 = ["Y", "N"]

label_encoder.fit(input_class)


gender = gender.iloc[:].values
gender = list(gender)
gender = label_encoder.transform(gender)

label_encoder.fit(input_class1)

status = status.iloc[:].values
employ = employ.iloc[:].values
employ = list(employ)
status = list(status)
status = label_encoder.transform(status)
employ = label_encoder.transform(employ)


label_encoder.fit(input_class2)

P_area = P_area.iloc[:].values
P_area = list(P_area)
P_area = label_encoder.transform(P_area)

label_encoder.fit(input_class3)

education = education.iloc[:].values
education = list(education)
education = label_encoder.transform(education)

label_encoder.fit(input_class4)

loan_status = loan_status.iloc[:].values
loan_status =list(loan_status)
loan_status = label_encoder.transform(loan_status)


# In[11]:


loan['Education'],loan['Self_Employed'],loan['Property_Area'],loan["Gender"],loan["Married"], loan['Loan_Status'], loan['Dependents'] =education, employ, P_area, gender, status, loan_status, dependat


# In[14]:


loan.head()
loan.to_csv("Loan Processed data.csv")


# In[15]:


data = loan.iloc[:, 1:-1]
target = loan.iloc[:, -1:]


# In[25]:


data.head(5)


# In[17]:


target.head(2)


# In[18]:


data.to_csv("Data.csv")
target.to_csv("Target.csv")


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 1/5, random_state=10)


# In[22]:


model_a = RandomForestClassifier()
model_a.fit(x_train, y_train)
pred = model_a.predict(x_test)
forest = accuracy_score(y_test, pred)



from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
k = accuracy_score(y_test, pred)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
pred = model.predict(x_test)
MB= accuracy_score(y_test, pred)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
pred = model.predict(x_test)
GB=accuracy_score(y_test, pred)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
LR=accuracy_score(y_test, pred)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
DTC=  accuracy_score(y_test, pred)

print ("""The result were as Follows\nRandom Forest Classifer {}\n\n K Nearest Neighbors {}
\n\nMultinomial NB {}\n\nGaussian NB {}\n\nLogistic Regression {}\n\n Decision Tree Classifier {}
""".format(forest,k, MB, GB, LR, DTC))


# In[23]:


from sklearn.externals import joblib


# In[24]:


joblib.dump(model_a, "Forest.pkl")


# In[28]:


n= data.iloc[:,:].values


# In[32]:


m = n[0]
m = np.array([m])


# In[33]:


model.predict(m)

