
.. code:: ipython3

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

.. code:: ipython3

    #========Loading the dataset======
    loan = pd.read_csv('loan.csv')

.. code:: ipython3

    #==========Visualising the Dataset======
    loan.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <th>1</th>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>N</td>
        </tr>
        <tr>
          <th>2</th>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <th>3</th>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <th>4</th>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    len(loan)




.. parsed-literal::

    614



.. code:: ipython3

    """
    We have to check if the dataset contain null values so as we 
    can replace them with mean values to improve our classification 
    
    and 
    
    we have to Encode string with Numerical values such as Gender and Education
    and Loan status
    """

.. code:: ipython3

    #=======Checking the presence of Null Values=========
    
    #==== We use built in Function isnull() and sum()
    
    loan.isnull().sum()




.. parsed-literal::

    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64



.. code:: ipython3

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

.. code:: ipython3

    loan.columns




.. parsed-literal::

    Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
          dtype='object')



.. code:: ipython3

    loan['LoanAmount'],loan['Loan_Amount_Term'], loan['Credit_History'] = loan_d[0], loan_d[1], loan_d[2]

.. code:: ipython3

    loan.isnull().sum()




.. parsed-literal::

    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount            0
    Loan_Amount_Term      0
    Credit_History        0
    Property_Area         0
    Loan_Status           0
    dtype: int64



.. code:: ipython3

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
    

.. code:: ipython3

    loan['Education'],loan['Self_Employed'],loan['Property_Area'],loan["Gender"],loan["Married"], loan['Loan_Status'], loan['Dependents'] =education, employ, P_area, gender, status, loan_status, dependat

.. code:: ipython3

    loan.head()
    loan.to_csv("Loan Processed data.csv")

.. code:: ipython3

    data = loan.iloc[:, 1:-1]
    target = loan.iloc[:, -1:]

.. code:: ipython3

    data.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5849</td>
          <td>0.0</td>
          <td>180.5</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    target.head(2)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    data.to_csv("Data.csv")
    target.to_csv("Target.csv")

.. code:: ipython3

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 1/5, random_state=10)

.. code:: ipython3

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


.. parsed-literal::

    The result were as Follows
    Random Forest Classifer 0.7967479674796748
    
     K Nearest Neighbors 0.6422764227642277
    
    
    Multinomial NB 0.5365853658536586
    
    Gaussian NB 0.6504065040650406
    
    Logistic Regression 0.6991869918699187
    
     Decision Tree Classifier 0.7317073170731707
    
    

.. parsed-literal::

    C:\Users\Kalebu\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      
    C:\Users\Kalebu\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\Kalebu\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

.. code:: ipython3

    from sklearn.externals import joblib

.. code:: ipython3

    joblib.dump(model_a, "Forest.pkl")




.. parsed-literal::

    ['Forest.pkl']



.. code:: ipython3

    n= data.iloc[:,:].values

.. code:: ipython3

    m = n[0]
    m = np.array([m])

.. code:: ipython3

    model.predict(m)




.. parsed-literal::

    array([1], dtype=int64)


