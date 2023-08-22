import warnings
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
import click

# Filter out unnecessary warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#global variable
sequence_length=1 

def preprocess_data(data):
    """
    Preprocess the input data by adding features like Holidays and weekdays, 
    and converting the 'Date' column to datetime format.
    
    Parameters:
    - data: The input data DataFrame.
    
    Returns:
    - df: Preprocessed DataFrame.
    """
    df = data
    df['Date'] = pd.to_datetime(df['Date'])
    df['Holidays'] = df['Date'].apply(lambda x: x in holidays.US()).astype(int)
    df['days'] = df['Date'].dt.weekday
    df.drop(columns=['Product ID'], axis=1, inplace=True)
    return df

def predict_sales(model_loc,data, input_date="08-30-2023"):#dummy date
    """
    Predict sales up to the given date using a trained model.
    
    Parameters:
    - model_loc: The location of the saved LSTM model.
    - data: The location of the input data CSV file.
    - input_date: The date up to which the sales are to be predicted. Default is "08-30-2023".
    
    Returns:
    - df: DataFrame containing actual and predicted sales up to the input date.
    - input_date: The date up to which sales were predicted.
    - l: The last date in the original dataset before predictions began.
    """
    
    # Load the saved model
    model = tf.keras.models.load_model(model_loc)

    # Load the data (assuming the same path as in Training.py)
    df = pd.read_csv(data)
    df = preprocess_data(df)

    # Predict sales count up to the given date
    last_date = df['Date'].iloc[-1]
    input_date = pd.to_datetime(input_date)
    l=last_date
    while last_date < input_date:
        #  Predict the 'Sale Count' for the next day using the current sequence.
        sequence = df[-sequence_length:].drop(columns=['Date']).values
        #sequence = features_scaler.transform(sequence)  # Scale only the features
        data_scaler = joblib.load('data_scaler.pkl')
        # sequence=data_scaler.transform(sequence)
        prediction = model.predict(sequence.reshape(1, sequence_length, 3),verbose=0)
        #Move to the next day and generate its 'Holidays' and 'days' features.
        last_date += pd.Timedelta(days=1)
        new_data = {
            'Date': last_date,
            'Sale Count': prediction[0][0].astype(int),  # Use the predicted 'Sale Count'
            'Holidays': 1 if last_date in holidays.US() else 0,
            'days': last_date.weekday()
        }
        # 3. Append this data to the dataframe.
        data_scaler = joblib.load('data_scaler.pkl')

        df = df.append(new_data, ignore_index=True)
    return df ,input_date, l
@click.command()
@click.option('--input_date', '-e', required=True, type=str, help='End date for prediction in format YYYY-MM-DD')
def main(input_date):
    """
    Command-Line Interface(CLI) entry point. Predicts sales for two products up to a given date and prints the predictions.
    
    Parameters:
    - input_date: The date up to which sales are to be predicted. Passed via the command line.
    """
    
    # Define paths for data files
    data1="a/product_1_80_a1_copy.csv"
    data2="a/vending_sales_only_product_2_80.csv"
    
    # Predict sales for the first product up to the given date
    a, last_date,input_date=predict_sales("a/model_product1.keras",data1,input_date)
    # Predict sales for the second product up to the given date
    b,_,_=predict_sales("a/model_product2.keras",data2,input_date)
    
    # Configure pandas to display all rows and columns
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Print the predictions for the two products
    print(a.iloc[-(last_date-input_date).days-1 :])
    print(b.iloc[-(last_date-input_date).days-1 :])
    
# If the script is executed as the main module, run the main() function
if __name__ == '__main__':
    main()

