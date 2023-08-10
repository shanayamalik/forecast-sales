import pandas as pd
import numpy as np
import holidays
import tensorflow as tf
from tensorflow.keras.models import load_model
import click

SEQUENCE_LENGTH = 8

def preprocess_data(data):
	df = data
	df['Date'] = pd.to_datetime(df['Date'])
	df['Holidays'] = df['Date'].apply(lambda x: x in holidays.US(years=df['Date'].dt.year.unique().tolist())).astype(int)
	df['days'] = df['Date'].dt.weekday
	df.drop(columns=['Product ID'], axis=1, inplace=True)
	return df

#@click.command()
#@click.argument('input_date')
def predict_sales(input_date="08-11-2023"):
	# Load the saved model
	model = tf.keras.models.load_model('model_product1.keras') # Replace with the path to your saved model
    
	# Load the data (assuming the same path as in Training.py)
	df = pd.read_csv('/kaggle/input/vending-sales/vending_sales_product_1.csv') # Replace with the path to your CSV
	df = preprocess_data(df)
    
	# Predict sales count up to the given date
	last_date = df['Date'].iloc[-1]
	input_date = pd.to_datetime(input_date)
	while last_date < input_date:
    	#  Predict the 'Sale Count' for the next day using the current sequence.
    	sequence = df[-SEQUENCE_LENGTH:].drop(columns=['Date']).values
    	#sequence = features_scaler.transform(sequence)  # Scale only the features
    	data_scaler = joblib.load('data_scaler.pkl')
    	prediction = model.predict(sequence.reshape(1, SEQUENCE_LENGTH, 3))

    	#Move to the next day and generate its 'Holidays' and 'days' features.
    	last_date += pd.Timedelta(days=1)
    	new_data = {
        	'Date': last_date,
        	'Sale Count': prediction[0][0],  # Use the predicted 'Sale Count'
        	'Holidays': 1 if last_date in holidays.US(years=[last_date.year]) else 0,
        	'days': last_date.weekday()
    	}
    	#new_data=data_scaler.fit_transform([[prediction[0][0],1 if last_date in holidays.US(years=[last_date.year]) else 0,last_date.weekday()]])
    	# 3. Append this data to the dataframe.
    	df = df.append(new_data, ignore_index=True)
	return df
   
# 	while last_date < input_date:
#     	last_date += pd.Timedelta(days=1)
#     	new_data = {
#         	'Date': last_date,
#         	'Holidays': 1 if last_date in holidays.US(years=[last_date.year]) else 0,
#         	'days': last_date.weekday()
#     	}
#     	new_data['Sale Count'] = 0  # Placeholder value
#     	sequence = df[-SEQUENCE_LENGTH:].drop(columns=['Date']).values
#     	sequence = np.append(sequence, [[new_data['Sale Count'], new_data['Holidays'], new_data['days']]], axis=0)[-SEQUENCE_LENGTH:]
#     	data_scaler = joblib.load('data_scaler.pkl')
#     	sequence[:, :] = features_scaler.transform(sequence[:, 1:])
#     	prediction = model.predict(sequence.reshape(1, SEQUENCE_LENGTH, 3))
#     	new_data['Sale Count'] = prediction[0][0]
#     	df = df.append(new_data, ignore_index=True)
    

if __name__ == '__main__':
	a=predict_sales()
