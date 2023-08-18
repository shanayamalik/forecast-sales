# Updated script content for training.py without one-hot encoding
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import holidays
import warnings
warnings.filterwarnings("ignore")
def train(data):
    df = pd.read_csv(data)
    df['Date']=pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Holidays']=df['Date'].apply(lambda x:x in holidays.US()).astype(int)
    # Prepare data for training
    X = df[['Weekday', 'Holidays']]
    y = df['Sale Count']

    # Convert 'Weekday' to category type
    X['Weekday'] = X['Weekday'].astype('category').cat.codes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost regressor
    regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
    regressor.fit(X_train, y_train)
    # Evaluate the model
    y_pred = regressor.predict(X_test)
    y_pred = regressor.predict(X_train)
    return regressor,y_train,y_pred
    # Save the trained model
regressor1,y_test,y_pred=train("product_1_80_a1_copy.csv")
regressor2,_,_=train("vending_sales_only_product_2_80.csv")
joblib.dump(regressor1, 'adaboost_rbegressor_model_no_encoding1.pkl')
joblib.dump(regressor2, 'adaboost_rbegressor_model_no_encoding2.pkl')
