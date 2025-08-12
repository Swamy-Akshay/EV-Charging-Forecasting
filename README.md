# EV ChargeGuard: A Hybrid Deep Learning System for Predictive EV Charging Optimization

## Project Overview
This project introduces a robust, hybrid deep learning system designed to forecast energy consumption and charging duration for electric vehicles (EVs). It aims to optimize charging schedules and enhance energy management systems by leveraging a combined model of Gated Recurrent Units (GRU) and Extreme Gradient Boosting (XGBoost).

The model was trained on an augmented real-world dataset of over 10,000 EV charging sessions. This approach captures both the sequential dependencies of time-series data (GRU) and the complex, non-linear feature relationships (XGBoost), demonstrating superior accuracy compared to traditional models.

## Key Features
-   Hybrid GRU + XGBoost architecture for enhanced accuracy.
-   Time-series prediction of Energy Consumption (in kWh) and Charging Duration (in hours).
-   Performance validation using key metrics: MAE, RMSE, MAPE, and R² Score.
-   Visual analytics for comparing predicted vs. actual results.

## Results Summary
| Metric      | GRU-XGBoost (Energy) | GRU-XGBoost (Time) |
|-------------|----------------------|--------------------|
| MAE         | 5.35 kWh             | 0.037 hours        |
| RMSE        | 7.70 kWh             | 0.056 hours        |
| MAPE        | 0.32%                | 0.02%              |
| R² Score    | 0.8754               | 0.9970             |

## Project Files
-   `GRU-XGBoost.ipynb`: Main Jupyter notebook for the hybrid model implementation and training.
-   `Model_Comparision.ipynb`: Notebook for benchmarking and comparing model performance.
-   `ev_charging_patterns_augmented.csv`: The dataset used for training and testing the models.
-   `akshay_report.docx`: Full project documentation.
-   `README.md`: Project overview and documentation.

## How to Run
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Swamy-Akshay/EV-Charging-Forecasting.git](https://github.com/Swamy-Akshay/EV-Charging-Forecasting.git)
    cd EV-Charging-Forecasting
    ```
2.  **Run the Notebooks:**
    Open `GRU-XGBoost.ipynb` in your preferred environment (Jupyter, VS Code, Google Colab) and execute the cells to reproduce the model training and results.

## Requirements
* Python 3.7+
* TensorFlow, Keras
* XGBoost, Scikit-learn
* Pandas, NumPy, Matplotlib, Seaborn