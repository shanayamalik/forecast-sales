import pandas as pd
import holidays
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

def create_sequence_data(data, sequence_len):
	x,y=[],[]
	for i in range(len(data[:,1])-sequence_len):
    	x.append(data[i:i+sequence_len])
    	y.append(data[:,0][i+sequence_len])
	return np.asarray(x), np.asarray(y)
def data_processing(data,sequence_length):

	df=data
	df['Date']=pd.to_datetime(df['Date'])
	df['Holidays']=df['Date'].apply(lambda x:x in holidays.US()).astype(int)
	df['days']=df['Date'].dt.weekday
    
	df.drop(columns=['Date','Product ID'],axis=1,inplace=True)
	data_scaler=sklearn.preprocessing.MinMaxScaler()
	df=data_scaler.fit_transform(df)
    
	train, test=train_test_split(df,test_size=0.2, shuffle=False)
	x_train , y_train=create_sequence_data(train,sequence_length)
	x_test , y_test=create_sequence_data(test,sequence_length)
    
	return x_train, y_train, x_test, y_test
    
def training(data,sequence_length):
    
	x_train, y_train, x_test, y_test=data_processing(data, sequence_length)
    
	EarlyStopping_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
	model_product1=Sequential([
    	SimpleRNN(100,activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])),
    	Dense(1)
	])
    
	model_product1.compile(optimizer='adam',loss='mse')

	  

history=model_product1.fit(x_train,y_train,epochs=250,validation_data=(x_test,y_test),
callbacks=[EarlyStopping_1])

return model_product1
    

df=pd.read_csv('/kaggle/input/vending-sales/vending_sales_product_1.csv')
sequence_length=8

model_product1=training(df,sequence_length)
