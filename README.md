# Amazon Stock Prediction Using LSTMs
This repository contains code for predicting the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs. The prediction is made using Tensorflow.

## Dataset
The dataset used in this project is obtained from the following Kaggle link: https://www.kaggle.com/datasets/kannan1314/amazon-stock-price-all-time. It contains daily stock prices of Amazon from 1997 to 2021.

## Requirements
* Python 3.10.2
* Pandas 1.3.5
* Numpy 1.23.3
* Tensorflow 2.11.0
* Keras 2.11.0
* Matplotlib 3.6.0


## Getting Started

### Clone the repository:
```
git clone https://github.com/Maximo-Rulli/Stock_price_prediction_LSTM
```


### Install the required packages:
```
pip install -r requirements.txt
```


### To run the graph of the final model versus the actual values run the following line:
```
python stock_view.py
```

## Approach
1. Preprocessed the dataset by scaling the features.
2. Created a time series dataset.
2. Divided the dataset into training and testing sets.
2. Built an LSTM model using Tensorflow.
2. Trained the model on the training dataset.
2. Evaluated the model on the testing dataset.
2. Made predictions on the test dataset.
2. Plotted the predictions against the actual values.

## Results
The LSTM model achieved a Mean Squared Error (MSE) of 0.0659 on the test dataset. The predicted values were plotted against the actual values, and the plot can be seen in the notebook.

## Conclusion
In this project, we successfully predicted the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs and Tensorflow. The model achieved a good MSE score on the validation dataset, however a remarkable difference is observed in the last 300 days, indicating that the model has not seen data similar to that beforehand and so it fails to accurately predict the stock values. Nonetheless it remains clear that despite the fact that there is a gap between the predicted and the actual values, the model still correctly predicts the tendency of the stock.
