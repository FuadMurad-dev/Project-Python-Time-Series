# Project-Python-Time-Series

A Python project for time series analysis and forecasting using `statsmodels` and `pmdarima`. The script loads a monthly dataset, visualizes the series, performs seasonal decomposition, trains ARIMA and Holt-Winters models, compares forecast accuracy, and generates a 12-month forecast.

## Features

- Load monthly time series data from `dataset.csv`
- Clean and convert data to a proper datetime index
- Plot the original series
- Perform additive seasonal decomposition
- Train a seasonal ARIMA model using `auto_arima`
- Train a Holt-Winters exponential smoothing model
- Compare model accuracy on a 12-month test split
- Generate a 12-month future forecast

## Requirements

- Python 3.8+
- pandas
- matplotlib
- numpy
- statsmodels
- pmdarima

## Installation

```bash
pip install pandas matplotlib numpy statsmodels pmdarima
```

## Usage

1. Place your monthly dataset in `dataset.csv`.
2. Make sure the dataset has two columns: a monthly date column and a value column.
3. Run the script:

```bash
python main.py
```

## Dataset Format

- The script expects `dataset.csv` to use `;` as the delimiter.
- The date column should use the `MM/YYYY` format.
- The value column should use a decimal comma (e.g. `1.234,56`) and will be converted to a decimal point format.

Example:

```csv
Data;Valor
01/2020;1.234,56
02/2020;1.250,00
```
