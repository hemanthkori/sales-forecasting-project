"""
Forecast daily sales using a seasonal ARIMA model.

This script reads a CSV file containing two columns: ``date`` and
``sales``.  If the file ``data/sample_sales.csv`` does not exist, a
synthetic dataset covering two years of daily sales is generated and
written to that path.  The script then fits a seasonal ARIMA model
with yearly seasonality to the sales series and produces a 30‑day
forecast.  Results are written to ``output/forecast.csv`` with
columns ``date`` and ``forecast``.  The script can optionally
generate a plot comparing the historical series and forecast when
invoked with ``--plot``.

Usage:

    python forecast.py [--input PATH] [--output PATH] [--plot]

Dependencies:
    pandas, numpy, matplotlib, statsmodels
"""

import argparse
from datetime import timedelta
import os
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_synthetic_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Generate two years of synthetic daily sales data and save to CSV.

    The data exhibits yearly seasonality and an upward trend with
    Gaussian noise.  The resulting DataFrame has two columns: ``date``
    and ``sales``.

    Parameters
    ----------
    csv_path : pathlib.Path
        Where to write the CSV.

    Returns
    -------
    pd.DataFrame
        The generated data frame.
    """
    rng = np.random.default_rng(42)
    start_date = pd.to_datetime("2023-01-01")
    periods = 365 * 2
    dates = pd.date_range(start_date, periods=periods, freq='D')
    # Create seasonal component: yearly seasonality using sine wave
    days = np.arange(periods)
    seasonal = 10 * np.sin(2 * np.pi * days / 365)
    trend = 0.05 * days  # small upward trend
    noise = rng.normal(0, 2, size=periods)
    sales = 100 + seasonal + trend + noise
    df = pd.DataFrame({"date": dates, "sales": sales})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load sales data from CSV, generating synthetic data if needed."""
    if not path.exists():
        print(f"Input file {path} not found; generating synthetic dataset.")
        df = generate_synthetic_data(path)
    else:
        df = pd.read_csv(path)
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df


def fit_and_forecast(series: pd.Series, steps: int = 30):
    """Fit a seasonal ARIMA model and forecast the specified number of steps.

    This uses SARIMAX with order (1,1,1) and seasonal order (1,1,1,365),
    which captures yearly seasonality.  Feel free to adjust these
    parameters for your own data.

    Parameters
    ----------
    series : pd.Series
        The time series of sales values indexed by date.
    steps : int, optional
        Number of forecast periods to produce. Default is 30.

    Returns
    -------
    pd.Series
        The forecast values indexed by date.
    """
    # Because yearly seasonality with daily data can be heavy to fit,
    # we limit the maximum number of parameters for quick convergence.
    # To ensure the model fits quickly on relatively small sample sizes,
    # we use weekly seasonality (period=7).  You can adjust the
    # seasonal order (p, d, q, m) to better fit your data.  A yearly
    # period (m=365) can be used when you have sufficient data, but
    # for demonstration purposes we opt for weekly seasonality.
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)
    return forecast


def save_forecast(forecast: pd.Series, output_path: pathlib.Path):
    """Save the forecast series to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = forecast.reset_index()
    df_out.columns = ['date', 'forecast']
    df_out.to_csv(output_path, index=False)


def plot_forecast(series: pd.Series, forecast: pd.Series, plot_path: pathlib.Path):
    """Generate a plot showing the recent history and forecast."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    # Plot last 60 days of history
    if len(series) > 60:
        history = series[-60:]
    else:
        history = series
    plt.plot(history.index, history.values, label='Historical Sales')
    # Forecast line begins after last date in history
    plt.plot(forecast.index, forecast.values, label='Forecast', color='tab:red')
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sales forecasting using SARIMA")
    parser.add_argument('--input', type=str, default='data/sample_sales.csv', help='Path to input CSV')
    parser.add_argument('--output', type=str, default='output/forecast.csv', help='Path to output forecast CSV')
    parser.add_argument('--plot', action='store_true', help='Generate a PNG plot of the forecast')
    args = parser.parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    # Load data
    df = load_data(input_path)
    series = df['sales']
    # Fit model and forecast
    fc = fit_and_forecast(series, steps=30)
    # Save forecast
    save_forecast(fc, output_path)
    # Plot if requested
    if args.plot:
        plot_path = pathlib.Path('plots/forecast_plot.png')
        plot_forecast(series, fc, plot_path)
        print(f"Plot saved to {plot_path}")
    print(f"Forecast saved to {output_path}")


if __name__ == '__main__':
    main()