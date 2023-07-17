#!/usr/bin/env python
# coding: utf-8

# # KRUTIKA .D. NAIDU

# # STOCK MARKET ANALYSIS & FORECASTING USING STACKED LSTM

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"C:/Users/Krutika/Desktop/TATAGLOBAL.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df1 = df.reset_index()['Close']


# In[6]:


df1.shape


# In[7]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[8]:


import numpy as np


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler((feature_range:=(0,1)))
df1 = scale.fit_transform(np.array(df1).reshape(-1,1))


# In[10]:


df1


# In[11]:


traning_size = int(len(df1)*0.70)
test_size = len(df1) - traning_size
train_data,test_data = df1[0:traning_size],df1[traning_size:len(df1)]


# In[12]:


traning_size,test_size


# In[13]:


train_data


# In[14]:


import numpy
def create_dataset(dataset,time_step=1):
    dataX,dataY = [],[]
    for i in range (len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)


# In[15]:


#reshape the sizes
time_step=100
X_train,Y_train = create_dataset(train_data,time_step)
X_test,Y_test = create_dataset(test_data,time_step)


# In[16]:


#reshape input as (sample,timestep,features)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[27]:


print(X_train.shape),print(Y_train.shape)


# In[28]:


## Create LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[32]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[33]:


model.summary()


# In[34]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)


# In[35]:


#Performance and check of the matices
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[38]:


train_predict=scale.inverse_transform(train_predict)
test_predict=scale.inverse_transform(test_predict)


# In[40]:


#Calculate the root mean square error performance
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))


# In[41]:


math.sqrt(mean_squared_error(Y_test,test_predict))


# In[44]:


#Plotting train predictions
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
#shift test  predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:,:]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:] = test_predict
#plot the baseline and predictions
plt.plot(scale.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[45]:


len(test_data)


# In[47]:


x_input = test_data[511:].reshape(1,-1)
x_input.shape


# In[48]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[49]:


temp_input


# In[54]:


from numpy import array

lst_output=[]
n_steps=101
i=0
while(i<30):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        #print x_input
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print temp_input
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)


# In[55]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[56]:


import matplotlib.pyplot as plt


# In[57]:


len(df1)


# In[58]:


df3=df1.tolist()
df3.extend(lst_output)


# In[59]:


plt.plot(day_new,scale.inverse_transform(df1[1935:]))
plt.plot(day_pred,scale.inverse_transform(lst_output))

