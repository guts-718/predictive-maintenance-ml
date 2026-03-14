# Predictive Maintenance using Machine Learning

This project explores how machine learning can be used to estimate the
Remaining Useful Life (RUL) of turbofan engines using sensor data.\
The goal is to detect degradation patterns and predict how long an
engine can continue operating before failure.

------------------------------------------------------------------------

## Project Idea

Modern industrial machines generate large amounts of sensor data.
Instead of performing maintenance on a fixed schedule or waiting for
failures to occur, we can analyze this data to estimate when a machine
is likely to fail.

This approach is called predictive maintenance.

In this project we train machine learning models that take engine sensor
readings as input and predict the remaining number of cycles before
failure.

------------------------------------------------------------------------

## Dataset

The project uses the NASA Turbofan Engine Degradation Dataset (C‑MAPSS).

This dataset simulates the behavior of turbofan engines as they
gradually degrade over time.

Each engine begins in a healthy condition and eventually fails.

### Data Structure

Each row in the dataset represents one operating cycle of an engine.

Columns include:

-   engine_id -- unique identifier for each engine
-   cycle -- operating cycle number
-   3 operational settings -- environmental conditions
-   21 sensor readings -- measurements from different engine components

So each row contains roughly 26 variables describing the state of the
engine at a particular time.

------------------------------------------------------------------------

## Target Variable

The prediction target is Remaining Useful Life (RUL).

RUL tells us how many cycles remain before the engine fails.

RUL is calculated as:

RUL = failure_cycle − current_cycle

To keep the learning problem stable, RUL values are clipped at 125
cycles.\
This prevents extremely large values early in the engine life from
dominating the training process.

------------------------------------------------------------------------

## Project Pipeline

The work is organized into several phases.

### 1. Data Preparation

-   Load the turbofan dataset
-   Assign proper column names
-   Compute Remaining Useful Life (RUL)
-   Clip RUL values
-   Inspect sensor correlations
-   Save cleaned dataset

Output: data/processed/train_clean.csv

------------------------------------------------------------------------

### 2. Sensor Analysis

The dataset contains sensors that are either constant or not
informative.

Steps performed:

-   Check sensor variance
-   Identify constant sensors
-   Visualize sensor trends across engine cycles
-   Normalize sensor values using standard scaling

Normalization formula:

z = (x − mean) / standard_deviation

Output: data/processed/train_scaled.csv

------------------------------------------------------------------------

### 3. Baseline Machine Learning Models

Simple models are trained first to establish a baseline.

Models used:

-   Linear Regression
-   Random Forest Regressor
-   Gradient Boosting Regressor

These models predict RUL using a single snapshot of sensor readings.

Evaluation metric:

Root Mean Squared Error (RMSE)

RMSE measures how far predicted RUL values are from the true values on
average.

------------------------------------------------------------------------

### Baseline Results

  Model              | RMSE
  -------------------| --------
  Linear Regression  | ~20.5
  Random Forest      | ~17.1
  Gradient Boosting  | ~17.1

Tree based models perform significantly better than linear regression
because the relationship between sensors and degradation is nonlinear.

------------------------------------------------------------------------

### 4. Feature Importance

Random Forest feature importance was used to identify which sensors
contribute most to predictions.

Some sensors dominate the signal, especially:

-   sensor_11
-   sensor_9
-   sensor_4

Low importance sensors were removed to reduce noise.

------------------------------------------------------------------------

## Key Observation

The current models treat each row independently:

sensor snapshot -\> predicted RUL

However engine degradation is a time dependent process.

Important information lies in how sensor values change over multiple
cycles, not just the current reading.

Because of this, classical models reach a performance limit.

------------------------------------------------------------------------

## Next Step

The next phase of the project introduces time series modeling.

Instead of predicting RUL from a single row, the model will receive a
sequence of sensor readings from multiple cycles.

Example:

cycles (t-29) ... (t) -\> predict RUL at cycle t

This allows deep learning models such as LSTM or GRU networks to learn
degradation trends.

-----------------------------------------------------------------------

## Sliding Window Experiment (Initial Results)

To introduce temporal context into the models, sliding windows of different sizes were generated from the sensor data. Each window represents a sequence of sensor readings across several engine cycles.

For classical machine learning models, each window was flattened into a feature vector and used to predict the Remaining Useful Life (RUL) at the last timestep of the window.

Window sizes tested:

{5, 10, 20, 30, 40}

Models evaluated:

- Linear Regression
- Random Forest
- XGBoost

### Results

| Window | LinearRegression | RandomForest | XGBoost |
|------|------|------|------|
| 5 | 21.17 | 17.89 | 17.85 |
| 10 | 20.50 | 18.04 | 17.76 |
| 20 | 18.66 | 17.99 | 16.15 |
| 30 | 17.41 | 17.98 | 14.97 |
| 40 | 16.95 | 17.49 | 13.71 |

### Observations

- Linear Regression improves as the window size increases, suggesting that temporal context helps even simple models.
- Random Forest performance remains relatively stable across window sizes.
- XGBoost shows significant improvement when larger windows are used.

## Issue Identified: Data Leakage

During analysis we discovered a potential data leakage issue in the current experimental setup.

Sliding windows were generated from the entire dataset and then split using a random train/test split:

train_test_split(X, y)

However, multiple windows originate from the same engine sequence. This means windows from the same engine may appear in both the training and testing sets.

Example:

train → engine1 cycles [1..30]  
test  → engine1 cycles [2..31]

Because these windows share overlapping sensor information, the model can indirectly learn patterns from the same engine in both training and testing, leading to overly optimistic evaluation results.

## Planned Fix

To remove this leakage, the data pipeline will be corrected in the next phase.

The corrected pipeline will be:

split engines first  
↓  
generate sliding windows per engine  
↓  
train models  

This ensures that all windows from a given engine appear exclusively in either the training or testing dataset.

A new notebook will implement this corrected pipeline and re-run the experiments with proper data separation.

--------------------------------------------------

## Repository Structure

## Project Structure

```
predictive-maintenance/
│
├── data
│   ├── processed
│   │   ├── train_clean.py
│   │   └── train_scaled.py
│   │
│   └── raw
│       └── train_FD001.txt
│
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_sensor_analysis.ipynb
│   └── 03_baseline_models.ipynb
│
├── pic_outputs
├── results
├── src
│
├── README.md
└── requirements.txt
```
------------------------------------------------------------------------

## Future Work

-   Sequence generation using sliding windows
-   LSTM and GRU models for time series prediction
-   Comparison with classical ML models
-   Model evaluation and visualization

------------------------------------------------------------------------

## Goal of the Project

The goal is to demonstrate how machine learning/ deep learning can be applied to
predictive maintenance by:

-   analyzing sensor degradation patterns
-   estimating remaining useful life of machines
-   comparing traditional ML methods with deep learning approaches
