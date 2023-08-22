import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import joblib
import holidays
import click
import warnings
warnings.filterwarnings("ignore")
# Load the trained models
regressor1 = joblib.load('adaboost_rbegressor_model_no_encoding1.pkl')
regressor2 = joblib.load('adaboost_rbegressor_model_no_encoding2.pkl')

# Load the original datasets to get the last dates
df1 = pd.read_csv('product_1_80_a1_copy.csv')
last_date_in_dataset1 = pd.to_datetime(df1['Date']).max()
df2 = pd.read_csv('vending_sales_only_product_2_80.csv')
last_date_in_dataset2 = pd.to_datetime(df2['Date']).max()

def predict_sales_for_product(end_date_obj, last_date_in_dataset, regressor):
    """
    Predict sales up to the given date using a trained model.
    
    Parameters:
    - end_date_obj: The date up to which the sales are to be predicted.
    - last_date_in_dataset: The last input data CSV file.
    - regressor: The saved LSTM model.
    
    Returns:
    - prediction: List containing predicted sales up to the end date.
    """
    # Generate date range from day after last date in dataset to provided end date
    date_range = pd.date_range(start=last_date_in_dataset + pd.Timedelta(days=1), end=end_date_obj)
    
    predictions = {}
    
    for date_obj in date_range:
        # Generate the 'Weekday' feature using dt.weekday and the 'Holiday' feature
        weekday = date_obj.weekday()
        holiday = 1 if date_obj in holidays.US() else 0

        # Prepare data for prediction
        X = pd.DataFrame({
            'Weekday': [weekday],
            'Holidays': [holiday]
        })
        # Convert 'Weekday' to category type
        X['Weekday'] = X['Weekday'].astype('category').cat.codes
        X1=[[weekday,holiday]]
        # Make the prediction
        prediction = regressor.predict(X1)
        
        # Store prediction
        predictions[date_obj.strftime('%Y-%m-%d')] = prediction[0].astype(int)
    
    return predictions

@click.command()
@click.option('--end_date', '-e', required=True, type=str, help='End date for prediction in format YYYY-MM-DD')
def main(end_date="2023-09-01"):
    """
    Command-Line Interface(CLI) entry point. Predicts sales for two products up to a given date and prints the predictions.
    
    Parameters:
    - input_date: The date up to which sales are to be predicted. Passed via the command line.
    """
    
    end_date_obj = pd.to_datetime(end_date)
    
    today = pd.Timestamp.now().normalize()  # normalize to get date without time
    days_difference = (today - end_date_obj).days
    if abs(days_difference)> 15:
        print("More than 15 days")
        return

    # Predict for Product 1
    predicted_sales1 = predict_sales_for_product(end_date_obj, last_date_in_dataset1, regressor1)
    print("Predictions for Product 1:")
    for date, sales in predicted_sales1.items():
        print(f"{date}: {sales}")
    
    print("\nPredictions for Product 2:")
    # Predict for Product 2
    predicted_sales2 = predict_sales_for_product(end_date_obj, last_date_in_dataset2, regressor2)
    for date, sales in predicted_sales2.items():
        print(f"{date}: {sales}")

if __name__ == '__main__':
    import pandas as pd
    main()
