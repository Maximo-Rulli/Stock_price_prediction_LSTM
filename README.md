# Amazon Stock Prediction Using LSTMs
This repository contains code for predicting the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs. The prediction is made using Tensorflow.

## Dataset
The dataset used in this project is obtained from the following Kaggle link (https://www.kaggle.com/datasets/kannan1314/amazon-stock-price-all-time). It contains daily stock prices of Amazon from 1997 to 2021.

## Requirements
* Python 3.x
* Tensorflow
* Pandas
* Numpy
* Matplotlib


## Getting Started

### Clone the repository:
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
1. Preprocessed the dataset by scaling the features.
2. Created a time series dataset.
2. Divided the dataset into training and testing sets.
2. Built an LSTM model using Tensorflow.
2. Trained the model on the training dataset.
2. Evaluated the model on the testing dataset.
2. Made predictions on the test dataset.
2. Plotted the predictions against the actual values.

## Results
The LSTM model achieved a Root Mean Squared Error (RMSE) of 0.0659 on the test dataset. The predicted values were plotted against the actual values, and the plot can be seen in the notebook.

## Conclusion
In this project, we successfully predicted the closing value of Amazon's stocks for the last 6000 days (aprox.) using LSTMs and Tensorflow. The model achieved a good RMSE score, indicating its effectiveness in predicting stock prices. This project can be extended to predict the stock prices of other companies as well.
