#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os


# In[2]:


# READING ALL THE CSV FILES IN files

os.chdir("C:/Users/srush/Documents/Python/Assignmnet2/Assignment 2 - Question/DS")
files = glob.glob('data*.csv')


# In[3]:


# FUNCTION TO TRANSFORM EVERY csv FILE DATA AND ADD TO NEW DATAFRAME

def transformData(temp):
    rows = temp.shape[0]
    cols = temp.shape[1]
  
    # Array to create every row of final dataframe to use as input
    rowValues = []
    for i in range(rows):
        for j in range(cols):
            val = temp.iloc[i,j]
            # Adding every non-null value to rowValue
            if val != None and type(val) ==  str:
                rowValues.append(val)
                
    return rowValues
  


# In[4]:


# FUNCTION TO PREPROCESS DATA AND TRANSFORM IT INTO ROW COLUMN FORMAT

def PreProcessData(df):   
    # Removing special characters from data
    df=df.replace('\[','',regex=True).astype(str)
    df=df.replace('\]','',regex=True).astype(str)
    df.columns = ["Values"]
    
    # Splitting every matrix value into separate column of dataframe
    temp = pd.DataFrame(df['Values'].str.split(' ').tolist())
    temp = temp.replace('',np.nan)
    # Generating each row of final dataframe
    row = transformData(temp)
    return row
    


# In[5]:


# CODE FOR CREATING THE FINAL DATAFRAME TO USE AS INPUT TO OUR MODEL

data = pd.DataFrame()
optimizedVals = []
for file in files:
    df = pd.read_csv(file)
    
    # Storing optimized values from all files
    optimizedVals.append(df.columns[0].split(':')[1])
    
    row = PreProcessData(df)
    row = pd.Series(row)
    #Adding every generated row into final dataframe
    data = data.append(row, ignore_index=True)
    print(data)
    
# Adding the class label column i.e Optimized Value for prediction    
data = data.assign(OptimVal = optimizedVals)
    


# In[6]:


data.head()


# In[55]:


#CREATING X and Y VARIABLES

X = data[data.columns[0:2499]].values
y = data["OptimVal"].values
#Converting string to numeric value
y = pd.to_numeric(y)


# In[56]:


# FUNCTION FOR SCALING AND NORMALIZING THE DATA USING MIN-MAX SCALER

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Normalizing data and creating the splits
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[53]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# In[29]:


from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Dropout, Flatten, MaxPooling1D, BatchNormalization


# In[11]:


# ANN MODEL CREATION

modelANN = Sequential()

modelANN.add(Dense(100,activation='relu',input_dim=X.shape[1]))
modelANN.add(Dense(75,activation='relu'))
modelANN.add(Dense(50,activation='relu'))
modelANN.add(Dense(25,activation='relu'))

modelANN.add(Dense(1))

# Using adam as optimizer and mean squared error as loss function
modelANN.compile(optimizer='adam',loss='mse')


# In[32]:


# CNN MODEL CREATION

modelCNN = Sequential()

# Model with 2 convolution layers and 3 fully connected layers
modelCNN.add(Conv1D(filters=100,kernel_size=3,activation='relu',input_shape = (X_train.shape[1],1)))
modelCNN.add(BatchNormalization())
modelCNN.add(MaxPooling1D(pool_size=2))
modelCNN.add(Conv1D(filters=50, kernel_size=3, activation='relu'))
modelCNN.add(BatchNormalization())
modelCNN.add(MaxPooling1D(pool_size=2))
modelCNN.add(Flatten())
modelCNN.add(Dense(100, activation='relu'))
modelCNN.add(Dense(50, activation='relu'))
modelCNN.add(Dense(50, activation='relu'))
modelCNN.add(Dense(1))

# Using adam as optimizer and mean squared error as loss function
modelCNN.compile(optimizer = 'adam', loss = 'mse')


# In[13]:


# USING CROSS VALIDATION FOR TRAINING THE ANN MODEL

from sklearn.model_selection import KFold

# Cross validation with 3 folds
crossVal = KFold(n_splits=3, shuffle = True , random_state=10)

# Training parameters are 200 epochs and 64 batch size
for trainidx, testidx in crossVal.split(X_train,y_train):
  X_train, X_validation = X[trainidx], X[testidx]
  y_train, y_validation = y[trainidx], y[testidx]
  modelANN.fit(X_train, y_train, validation_data = (X_validation, y_validation),epochs=200,batch_size=64)
  predictANN = modelANN.predict(X_validation)


# In[33]:


# Expanding input dimension to match with the CNN model input
X = np.expand_dims(X, axis=2)


# In[34]:


# USING CROSS VALIDATION FOR TRAINING THE ANN MODEL

from sklearn.model_selection import KFold

# Cross validation with 3 folds
crossVal = KFold(n_splits=3, shuffle = True , random_state=101)

# Training parameters are 100 epochs and 128 batch size
for trainidx, testidx in crossVal.split(X_train,y_train):
  X_train, X_validation = X[trainidx], X[testidx]
  y_train, y_validation = y[trainidx], y[testidx]
  history = modelCNN.fit(X_train, y_train, 
                        validation_data = (X_validation, y_validation), 
                        epochs=100, 
                        batch_size=128)
  predictCNN = modelCNN.predict(X_validation)


# In[57]:


# LOADING THE SAVED MODEL

from tensorflow.keras.models import load_model
modelANN = load_model('C:/Users/srush/Documents/Python/Assignmnet2/Assignment 2 - Question/my_model_ANN_1102484.h5')
modelCNN = load_model('C:/Users/srush/Documents/Python/Assignmnet2/Assignment 2 - Question/my_modelCNN_1102484.h5')



# ****** Uncomment the below chunk if the prediction chunks throws error ****** #

#X = data[data.columns[0:2499]].values
#y = data["OptimVal"].values
#y = pd.to_numeric(y)
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# ***************************************************************************** #

# In[58]:

# PREDICTION WITH THE TRAINED/SAVED MODEL

predictionsANN = modelANN.predict(X_test)

X_test = np.expand_dims(X_test, axis=2)
predictionsCNN = modelCNN.predict(X_test)


# In[77]:


# FUNCTION TO CALCULATE THE ERROR VALUE AND SHOW PREDICTION PERFORMANCE FOR BOTH THE MODELS

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

print("Mean Absolute Error for ANN:",mean_absolute_error(y_test,predictionsANN))
print("Root Mean Square Error for ANN:",np.sqrt(mean_squared_error(y_test,predictionsANN)))
print("Mean Absolute Error for CNN:",mean_absolute_error(y_test,predictionsCNN))
print("Root Mean Square Error for CNN:",np.sqrt(mean_squared_error(y_test,predictionsCNN)))


# In[74]:


# FUNCTION FOR COMPARISION FOR PREDICTIONS USING ANN AND CNN

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)

# Model predictions
plt.scatter(y_test,predictionsANN)
# Perfect predictions
plt.plot(y_test,y_test,'red')
plt.title('Comparision for ANN')
plt.legend(['Actual Value','Predicted Value'], loc='lower right')

plt.subplot(1,2,2)
# Model predictions
plt.scatter(y_test,predictionsCNN)
#Perfect predictions
plt.plot(y_test,y_test,'red')
plt.title('Comparision for CNN')
plt.legend(['Actual Value','Predicted Value'], loc='lower right')
plt.show()


# In[70]:


# FUNCTION FOR COMPARISION THE PERFORMANCE OF PREDICTIONS USING ANN AND CNN

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)

plt.plot(predictionsANN)
plt.plot(y_test)
plt.title('Performance For ANN')
plt.ylabel('Optimized Value')
plt.xlabel('Predictions')
plt.legend(['Predicted Value', 'Actual Value'], loc='lower right')

plt.subplot(1,2,2)
plt.plot(predictionsCNN)
plt.plot(y_test)
plt.title('Performance for CNN')
plt.ylabel('Optimized Value')
plt.xlabel('Predictions')
plt.legend(['Predicted Value', 'Actual Value'], loc='lower right')
plt.show()

# ACCORDING TO THE EXPERIMENT, CNN PERFORMED BETTER COMPARED TO ANN
# RMSE FOR CNN : 43.92
# RMSE FOR ANN : 53.88

