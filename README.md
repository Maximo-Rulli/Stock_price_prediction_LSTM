# Amazon Stock Prediction Using LSTMs
This repository contains code for predicting the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs. The prediction is made using Tensorflow.

## Dataset
The dataset used in this project is obtained from Yahoo Finance. It contains daily stock prices of Amazon from 2001 to 2021.

## Requirements
Python 3.x
Tensorflow
Pandas
Numpy
Matplotlib
Getting Started
Clone the repository:

```
git clone https://github.com/your-username/amazon-stock-prediction.git
```


### Install the required packages:
```
pip install -r requirements.txt
```


### Run the Jupyter notebook amazon_stock_prediction.ipynb:
```
jupyter notebook amazon_stock_prediction.ipynb
```

## Approach
Preprocessed the dataset by scaling the features.
Created a time series dataset.
Divided the dataset into training and testing sets.
Built an LSTM model using Tensorflow.
Trained the model on the training dataset.
Evaluated the model on the testing dataset.
Made predictions on the test dataset.
Plotted the predictions against the actual values.

## Results
The LSTM model achieved a Root Mean Squared Error (RMSE) of 0.0659 on the test dataset. The predicted values were plotted against the actual values, and the plot can be seen in the notebook.

## Conclusion
In this project, we successfully predicted the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs and Tensorflow. The model achieved a good RMSE score, indicating its effectiveness in predicting stock prices. This project can be extended to predict the stock prices of other companies as well.
