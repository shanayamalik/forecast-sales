import pandas as pd
import holidays
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import joblib
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import random
import holidays
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def inverse_scale_single_array(scaler,single_array):
    # scaler = joblib.load('data_scaler.pkl')
    reshaped_array = np.array(single_array).reshape(-1, 1)
    zeros = np.zeros((reshaped_array.shape[0], 2))
    combined_data = np.hstack((reshaped_array, zeros))
    actual_pred=scaler.inverse_transform(combined_data)
    return actual_pred[:,0]
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
    df=data_scaler.inverse_transform(df)
    joblib.dump(data_scaler, 'data_scaler.pkl')

    train, test=train_test_split(df,test_size=0.2, shuffle=False,random_state=42)
    x_train , y_train=create_sequence_data(train,sequence_length)
    x_test , y_test=create_sequence_data(test,sequence_length)
    return x_train, y_train, x_test, y_test

def training2(data,sequence_length):

    x_train, y_train, x_test, y_test=data_processing(data, sequence_length)

    EarlyStopping_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    model_product1=Sequential([
        LSTM(50,activation='relu',return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])),
        LSTM(128,activation='relu'),
        Dense(1)
    ])

    model_product1.compile(optimizer='adam',loss='mse')

    history=model_product1.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),callbacks=[EarlyStopping_1],verbose=0)
    # Convert the history to a DataFrame
    df = pd.DataFrame(history.history)

    # Save the DataFrame to CSV
    df.to_csv('training_history.csv', index=False)
    return model_product1

def training1(data,sequence_length):

    x_train, y_train, x_test, y_test=data_processing(data, sequence_length)

    EarlyStopping_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model_product1=Sequential([
        LSTM(50,activation='relu',return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])),
        LSTM(128,activation='relu'),
        Dense(1)
    ])
    model_product1.compile(optimizer='adam',loss='mae')

    history=model_product1.fit(x_train,y_train,epochs=200,validation_data=(x_test,y_test),callbacks=[EarlyStopping_1],verbose=0)
    # Convert the history to a DataFrame
    df = pd.DataFrame(history.history)
    # Save the DataFrame to CSV
    df.to_csv('training_history.csv', index=False)
    return model_product1

import pandas as pd
data1="product_1_80_a1_copy.csv"
data2="a/vending_sales_only_product_2_80.csv"
df1=pd.read_csv("a/product_1_80_a1_copy.csv")
df2=pd.read_csv(data2)
sequence_length=1
model_product1 =training1(df1,sequence_length)
model_product1.save('a/model_product1.keras')
model_product2=training2(df2,sequence_length)
model_product2.save('a/model_product2.keras')
