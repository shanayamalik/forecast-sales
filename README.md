# Vending Machine Sales Forecasting

This repository provides solutions for forecasting sales of products in a vending machine using two different approaches: AdaBoost and LSTM.

## Directory Structure

    .
    
    ├── adaboost
    |   ├── models
    │   │   ├── Forecasting.pkl
    │   ├── dataset
    │   │   ├── product1.csv
    │   │   └── product2.csv
    │   ├── training.py
    │   └── inference.py
    ├── lstm
    |   ├── models
    │   │   ├── Forecasting.keras
    │   ├── dataset
    │   │   ├── product1.csv
    │   │   └── product2.csv
    │   ├── training.py
    │   └── inference.py
    └── LICENSE


## 📄 Dataset

Each product dataset spans 60 days and features the following columns:
- **Date**: The date of sales.
- **Sales**: The number of sales on that date.

Additional data attributes like holidays and weekends have been extrapolated based on the date.

## ⚙️ Usage

Both the AdaBoost and LSTM models have their respective directories, each with a similar structure and usage pattern.

### Training

1. **Preparation**: 
   - Before starting the training process, ensure that you have the necessary data in the `dataset` directory of either the `AdaBoost` or `LSTM` models. The data should be in the form of `.csv` files.

2. **Execution**: 
   - To initiate the training process for your chosen model:
     1. Navigate to the desired directory (`AdaBoost` or `LSTM`).
     2. Run the training script:
        ```bash
        python training.py
        ```
     - This script will process the data, train the model, and save the trained model in the `models` directory.

### Forecasting

After training, you can forecast future sales using the trained model.

1. **Preparation**: 
   - Ensure that the model has been trained and saved in the `models` directory.

2. **Execution**:
   1. Navigate to the directory of the model you want to use for forecasting (`AdaBoost` or `LSTM`).
   2. Run the inference script with the desired date:
      ```bash
      python inference.py --date YYYY-MM-DD
      ```
   - Replace `YYYY-MM-DD` with the date up to which you wish to forecast sales. The script will use the trained model to predict sales up to the specified date and display the results.

## License
This project is licensed under the terms mentioned in the LICENSE file.
