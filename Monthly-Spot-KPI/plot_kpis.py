"""
AWS Lambda Function: QuickSight Data Preparation with Incremental Loading

This function prepares and updates forecast data for QuickSight visualization.
Can run at any frequency: daily, weekly, monthly, etc.

Features:
- Automatic mode detection (Full Load vs Incremental)
- FULL LOAD (first run): Downloads all historical forecast and actual data
- INCREMENTAL (subsequent runs):
  * Smart cutoff: Uses last date from existing CSV to determine what to download
  * Works for ANY frequency: daily, weekly, monthly, etc.
  * Forecast: Only files modified after last CSV date (filters by S3 LastModified)
  * Actual: Complete CSV (one file updated daily, filters to new rows only)
- Efficient: Skips already-processed data
- Deduplication: Removes duplicate timestamps, keeping the latest values
- Rolling window: Maintains exactly 1 year of data in the output
- Model support: XGBoost, LightGBM, and LSTM forecasts
- KPI calculation: MAE, RMSE, SMAPE for each model

Outputs:
- latest_combined_forecast_actual.csv: Clean dataset with actual values and forecasts
- latest_daily_kpis.csv: Daily aggregated performance metrics
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pipeline.config import (
    DAYS_TO_KEEP,
    START_DATE,
    END_DATE,
    JST,
    get_model_names,
    get_model_display_name,
    setup_logging
)
from pipeline.s3_operations import S3Handler
from pipeline.data_loaders import DataLoader
from pipeline.data_processors import DataProcessor
from pipeline.metrics_calculator import MetricsCalculator

logger = setup_logging()

output_folder = f"KPI/{START_DATE}_to_{END_DATE}"

# Model-wise output directories (accumulated across pipeline runs)
LSTM_OUTPUT_DIR     = "KPI/lstm"
BOOSTING_OUTPUT_DIR = "KPI/boosting"


def _update_model_wise_dirs(period_folder: str):
    """
    After a pipeline run, copy/append the period data into the model-wise
    KPI/lstm/ and KPI/boosting/ directories so that analysis notebooks can
    load the full history from a single path.
    """
    import shutil

    os.makedirs(LSTM_OUTPUT_DIR,     exist_ok=True)
    os.makedirs(BOOSTING_OUTPUT_DIR, exist_ok=True)

    af = pd.read_csv(f"{period_folder}/actual_forecast.csv", parse_dates=["datetime"])
    dk = pd.read_csv(f"{period_folder}/daily_kpis.csv",      parse_dates=["date"])
    mk = pd.read_csv(f"{period_folder}/monthly_kpis.csv")

    boosting_cols = [c for c in af.columns if c.endswith("_forecast") and c != "lstm_forecast"]

    # ── LSTM model-wise directory ─────────────────────────────────────────────
    lstm_af_path = f"{LSTM_OUTPUT_DIR}/actual_forecast.csv"
    new_lstm_af  = af[["datetime", "actual_value", "lstm_forecast"]].copy()
    if os.path.exists(lstm_af_path):
        existing = pd.read_csv(lstm_af_path, parse_dates=["datetime"])
        new_lstm_af = pd.concat([existing, new_lstm_af], ignore_index=True)
    new_lstm_af = new_lstm_af.drop_duplicates(subset="datetime", keep="last").sort_values("datetime")
    new_lstm_af.to_csv(lstm_af_path, index=False)

    lstm_dk_path = f"{LSTM_OUTPUT_DIR}/daily_kpis.csv"
    new_lstm_dk  = dk[["date", "lstm_mae", "lstm_rmse", "lstm_smape"]].copy()
    if os.path.exists(lstm_dk_path):
        existing = pd.read_csv(lstm_dk_path, parse_dates=["date"])
        new_lstm_dk = pd.concat([existing, new_lstm_dk], ignore_index=True)
    new_lstm_dk = new_lstm_dk.drop_duplicates(subset="date", keep="last").sort_values("date")
    new_lstm_dk.to_csv(lstm_dk_path, index=False)

    lstm_mk_path = f"{LSTM_OUTPUT_DIR}/monthly_kpis.csv"
    new_lstm_mk  = mk[["month", "lstm_mae", "lstm_rmse", "lstm_smape"]].copy()
    if os.path.exists(lstm_mk_path):
        existing = pd.read_csv(lstm_mk_path)
        new_lstm_mk = pd.concat([existing, new_lstm_mk], ignore_index=True)
    new_lstm_mk = new_lstm_mk.drop_duplicates(subset="month", keep="last").sort_values("month")
    new_lstm_mk.to_csv(lstm_mk_path, index=False)

    # ── Boosting model-wise directory (only if boosting data is present) ──────
    if boosting_cols:
        for fname in ["actual_forecast.csv", "daily_kpis.csv", "monthly_kpis.csv"]:
            shutil.copy(f"{period_folder}/{fname}", f"{BOOSTING_OUTPUT_DIR}/{fname}")

    logger.info(f"Model-wise dirs updated: {LSTM_OUTPUT_DIR}, {BOOSTING_OUTPUT_DIR}")


def plot_daily_actual_vs_forecast(df: pd.DataFrame, output_folder: str):
    """
    Plot daily actual vs forecast values
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    plt.figure(figsize=(14, 6))
    plt.plot(df['datetime'], df['actual_value'], label='Actual', color='blue', linewidth=1)

    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for idx, model_name in enumerate(get_model_names()):
        forecast_col = f'{model_name}_forecast'
        if forecast_col in df.columns:
            plt.plot(df['datetime'], df[forecast_col],
                    label=f'{model_name.upper()} Forecast',
                    color=colors[idx % len(colors)],
                    linewidth=1, alpha=0.7)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Daily Actual vs Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plot_path = f"{output_folder}/daily_actual_vs_forecast.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved daily plot to: {plot_path}")


def plot_daily_kpis(daily_kpi_df: pd.DataFrame, output_folder: str):
    """
    Plot daily KPIs (MAE, RMSE, SMAPE)
    """
    df = daily_kpi_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    metrics = ['mae', 'rmse', 'smape']
    titles = ['MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)', 'SMAPE (%)']
    colors = ['red', 'green', 'orange', 'purple', 'brown']

    for ax, metric, title in zip(axes, metrics, titles):
        for idx, model_name in enumerate(get_model_names()):
            col = f'{model_name}_{metric}'
            if col in df.columns:
                ax.plot(df['date'], df[col],
                       label=model_name.upper(),
                       color=colors[idx % len(colors)],
                       linewidth=1.5)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plot_path = f"{output_folder}/daily_kpis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved daily KPIs plot to: {plot_path}")


def plot_monthly_actual_vs_forecast(df: pd.DataFrame, output_folder: str):
    """
    Plot monthly aggregated actual vs forecast values as bar chart
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.to_period('M').astype(str)

    # Aggregate by month
    agg_cols = {'actual_value': 'mean'}
    for model_name in get_model_names():
        forecast_col = f'{model_name}_forecast'
        if forecast_col in df.columns:
            agg_cols[forecast_col] = 'mean'

    monthly_df = df.groupby('month').agg(agg_cols).reset_index()

    # Plot
    x = range(len(monthly_df))
    width = 0.8 / (len(get_model_names()) + 1)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    plt.figure(figsize=(12, 6))

    # Plot actual
    plt.bar([i - width * len(get_model_names()) / 2 for i in x],
            monthly_df['actual_value'], width=width, label='Actual', color='blue', alpha=0.8)

    # Plot forecasts
    for idx, model_name in enumerate(get_model_names()):
        forecast_col = f'{model_name}_forecast'
        if forecast_col in monthly_df.columns:
            offset = (idx + 1 - len(get_model_names()) / 2) * width
            plt.bar([i + offset for i in x], monthly_df[forecast_col],
                   width=width, label=f'{model_name.upper()} Forecast',
                   color=colors[(idx + 1) % len(colors)], alpha=0.8)

    plt.xlabel('Month')
    plt.ylabel('Average Price')
    plt.title('Monthly Actual vs Forecast (Average)')
    plt.xticks(x, monthly_df['month'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plot_path = f"{output_folder}/monthly_actual_vs_forecast.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved monthly actual vs forecast plot to: {plot_path}")


def plot_monthly_kpis(monthly_kpi_df: pd.DataFrame, output_folder: str):
    """
    Plot monthly KPIs as bar charts
    """
    df = monthly_kpi_df.copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    metrics = ['mae', 'rmse', 'smape']
    titles = ['Monthly MAE', 'Monthly RMSE', 'Monthly SMAPE (%)']

    x = range(len(df))
    width = 0.8 / max(len(get_model_names()), 1)
    colors = ['red', 'green', 'orange', 'purple', 'brown']

    for ax, metric, title in zip(axes, metrics, titles):
        for idx, model_name in enumerate(get_model_names()):
            col = f'{model_name}_{metric}'
            if col in df.columns:
                offset = (idx - len(get_model_names())/2 + 0.5) * width
                ax.bar([i + offset for i in x], df[col],
                      width=width, label=model_name.upper(),
                      color=colors[idx % len(colors)], alpha=0.8)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df['month'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = f"{output_folder}/monthly_kpis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved monthly KPIs plot to: {plot_path}")


def plot_kpis_from(output_folder):
    """
    FUNCTION 2: Calculate KPIs from unified CSV

    - Loads unified CSV from S3
    - Recalculates error metrics
    - Calculates daily aggregated KPIs
    - Saves KPI CSV to S3

    Returns:
        dict: Summary with number of days, KPI file location
    """
    logger.info("=" * 60)
    logger.info("FUNCTION 2: CALCULATE KPIs")
    logger.info("=" * 60)

    # Generate plots
    logger.info("Generating plots...")
    # read csv from output folder actual vrs forecast plot and daily/monthly kpi plots
    unified_df = pd.read_csv(f"{output_folder}/actual_forecast.csv")
    daily_kpi_df = pd.read_csv(f"{output_folder}/daily_kpis.csv")
    monthly_kpi_df = pd.read_csv(f"{output_folder}/monthly_kpis.csv")

    # Create plots subfolder
    plots_folder = f"{output_folder}/plots"
    os.makedirs(plots_folder, exist_ok=True)

    plot_daily_actual_vs_forecast(unified_df, plots_folder)
    plot_monthly_actual_vs_forecast(unified_df, plots_folder)
    plot_daily_kpis(daily_kpi_df, plots_folder)
    plot_monthly_kpis(monthly_kpi_df, plots_folder)

    # Update model-wise KPI/lstm/ and KPI/boosting/ directories
    _update_model_wise_dirs(output_folder)

    logger.info("   Summary:")
    logger.info(f"   - Rows: {daily_kpi_df}")
    logger.info(f"   - Total days: {len(daily_kpi_df)}")
    logger.info(f"   - Date range: {daily_kpi_df['date'].min()} to {daily_kpi_df['date'].max()}")
    logger.info(f"   - KPI columns: {', '.join([col for col in daily_kpi_df.columns if col != 'date'])}")

    return {
        'days': len(daily_kpi_df),
        'date_range': {
            'start': str(daily_kpi_df['date'].min()),
            'end': str(daily_kpi_df['date'].max())
        },
    }


def lambda_handler(event, context):
    """Main Lambda handler - orchestrates both functions"""
    logger.info("=" * 60)
    logger.info("AWS Lambda: QuickSight Data Preparation")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)

    try:
        # Function 1: Calculate KPIs
        data_result = pd.read_csv(f"{output_folder}/actual_forecast.csv")
        plot_kpis_from(output_folder)
        # Print final summary
        logger.info("=" * 60)
        logger.info("SUCCESS - All processing complete!")
        logger.info("=" * 60)
        if START_DATE and END_DATE:
            logger.info(f"Date range filter: {START_DATE} to {END_DATE}")
        else:
            logger.info(f"Rolling window: {DAYS_TO_KEEP} days")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data preparation completed successfully'
            })
        }


    except Exception as e:
        logger.error(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing data',
                'error': str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    # Mock Lambda context
    class Context:
        function_name = "quicksight-data-prep"
        memory_limit_in_mb = 512
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:quicksight-data-prep"
        aws_request_id = "test-request-id"

    result = lambda_handler({}, Context())
    logger.info(f"Result: {result}")
