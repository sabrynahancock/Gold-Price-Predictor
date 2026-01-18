# Machine Learning App for Commodity Price Prediction

## Name and Purpose of the Application

The application is called **Gold Price Predictor**. Its purpose is to predict the future closing price of gold using a supervised machine learning model trained on historical commodity price data. This project demonstrates the complete lifecycle of an applied machine learning system, including data acquisition, preprocessing, feature engineering, model training, evaluation, and future price prediction. The goal of the application is to apply machine learning concepts learned in the course to a real-world financial forecasting problem.

---

## Algorithms Used

This application uses **Random Forest Regression**, a supervised machine learning algorithm based on an ensemble of decision trees. Random Forest was selected because it performs well on structured, tabular data and can model non-linear relationships between input features and the target variable. By combining multiple decision trees and averaging their predictions, the model helps reduce overfitting and improve prediction stability. This approach aligns with course topics related to decision tree algorithms and supervised learning.

---

## Dataset Information

### Dataset Source

The dataset used in this project consists of historical daily price data for **Gold Futures (GC=F)** obtained from **Yahoo Finance** using the `yfinance` Python library. The dataset includes open, high, low, close, adjusted close prices, and trading volume.

### Number of Records

After preprocessing and feature engineering, the dataset used by the application contained **2,755 records**. Each record represents one trading day of gold futures data with engineered features derived from historical prices.

### Number of Features

The model uses **11 engineered input features** created from historical price and volume data.

### Feature Description

| Feature Name | Description | Data Type |
|-------------|------------|-----------|
| close_lag1 | Closing price from the previous trading day | Numeric |
| close_lag2 | Closing price from two trading days ago | Numeric |
| ret_1d | Daily percentage return of the closing price | Numeric |
| ma_5 | 5-day moving average of the closing price | Numeric |
| ma_10 | 10-day moving average of the closing price | Numeric |
| ma_20 | 20-day moving average of the closing price | Numeric |
| vol_10 | 10-day rolling volatility of daily returns | Numeric |
| vol_20 | 20-day rolling volatility of daily returns | Numeric |
| hl_range | Daily high–low range normalized by close | Numeric |
| oc_change | Percentage change from open to close | Numeric |
| Volume | Daily trading volume | Numeric |

**Target Variable:**  
The target variable is the **next trading day’s closing price**, created by shifting the closing price forward by one day.

### Preprocessing Steps

1. Download historical gold futures data from Yahoo Finance.
2. Sort the data chronologically to preserve time-series order.
3. Create lag features, daily returns, moving averages, and rolling volatility indicators.
4. Generate the target variable representing the next-day closing price.
5. Remove rows containing missing values caused by rolling calculations.

---

## Libraries, Toolkits, and Frameworks

- **Python:** Core programming language used for the application.
- **Pandas & NumPy:** Data manipulation, transformation, and numerical operations.
- **yfinance:** Retrieval of historical gold futures price data.
- **Scikit-learn:** Model training and evaluation using RandomForestRegressor.
- **Matplotlib:** Visualization of predicted versus actual prices.

---

## Application Design and Implementation

The application follows a structured machine learning pipeline. First, historical gold price data is downloaded and cleaned. Feature engineering is then applied to extract relevant patterns from the time-series data. The dataset is split into training and testing sets using a time-based split to avoid data leakage. A Random Forest Regression model is trained on the training data and evaluated on unseen test data. Finally, the trained model is used to predict the next trading day’s gold price, and the results are printed to the console and visualized in a saved plot.

---

## Instructions for Running the Application

1. Create and activate a Python virtual environment.
2. Install the required dependencies listed in the project.
3. Run the application script.
4. Review the printed evaluation metrics, predicted price, and generated visualization.

---

## Results

When executed, the application produced a dataset containing **2,755 records** and **11 engineered features**. Model evaluation on the test dataset resulted in the following metrics:

- **Mean Absolute Error (MAE):** 884.03  
- **Root Mean Squared Error (RMSE):** 1,127.00  
- **R² Score:** -1.5990  

The application also generated a predicted closing price for the next trading day and saved a plot comparing predicted versus actual values for the test dataset.

---

## Discussion and Insights

The Random Forest Regression model provides a reasonable baseline for predicting gold prices using engineered historical features. However, gold prices are influenced by many external factors such as inflation, interest rates, geopolitical events, and currency fluctuations that are not included in this dataset. The negative R² score indicates that historical price-based features alone are insufficient to fully explain future price movements, which highlights the inherent difficulty of financial time-series forecasting. Future improvements could include incorporating macroeconomic indicators, experimenting with additional models such as Support Vector Machines or Neural Networks, and extending the model to predict prices over longer time horizons.

---

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Yahoo Finance. (n.d.). *Gold futures (GC=F) historical prices & data*. https://finance.yahoo.com

scikit-learn developers. (n.d.). *RandomForestRegressor documentation*. https://scikit-learn.org

ranaroussi. (n.d.). *yfinance documentation*. https://github.com/ranaroussi/yfinance
