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



def prepare_unified_data():
    """
    FUNCTION 1: Download and merge forecast + actual data

    - Checks if unified CSV exists in S3
    - If exists: Incremental mode (only new forecasts + recent actual)
    - If not: Full load mode (all historical data)
    - Merges, deduplicates, applies rolling window
    - Saves clean unified CSV to S3

    Returns:
        dict: Summary with rows, date range, mode
    """
    logger.info("=" * 60)
    logger.info("FUNCTION 1: PREPARE UNIFIED DATA")
    logger.info("=" * 60)

    # Initialize handlers
    s3_handler = S3Handler()
    data_loader = DataLoader(s3_handler)
    processor = DataProcessor()



    # Get current time in JST for logging
    now_jst = datetime.now(JST)
    now_utc = datetime.now(timezone.utc)

    # Parse date range from config
    if START_DATE and END_DATE:
        start = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        logger.info(f"Using date range: {START_DATE} to {END_DATE}")
    else:
        start = now_jst - timedelta(days=DAYS_TO_KEEP)
        end = now_jst
        logger.info(f"Using rolling window: last {DAYS_TO_KEEP} days")

    # Step 1: Load actual data
    total_steps = 2 + len(get_model_names())  # actual + models + merge
    logger.info(f"[1/{total_steps}] Loading actual data...")
    new_actual_df = data_loader.load_actual_data(modified_after=None)

    # Step 2-N: Load forecasts dynamically for all configured models
    forecast_dfs = {}
    model_names = get_model_names()
    for idx, model_name in enumerate(model_names, start=2):
        # Go one day back to ensure we don't miss files uploaded the day before
        fetch_after = start - timedelta(days=2)
        logger.info(f"[{idx}/{total_steps}] Loading {get_model_display_name(model_name)} forecasts (modified after {fetch_after.date()})...")
        model_df = data_loader.load_forecast_data(model_name, modified_after=fetch_after)

        if not model_df.empty:
            logger.info(f"   ✓ Loaded {len(model_df)} rows of {get_model_display_name(model_name)} forecast data")
        else:
            logger.info(f"   ✓ No new {get_model_display_name(model_name)} forecast data found")

        forecast_dfs[model_name] = model_df

    # Final Step: Merge and process datasets
    logger.info(f"[{total_steps}/{total_steps}] Merging and processing datasets...")
    if not new_actual_df.empty:
        new_unified_df = processor.merge_datasets(new_actual_df, forecast_dfs)
        logger.info(f"   ✓ Merged new dataset has {len(new_unified_df)} rows")

        unified_df = new_unified_df

        # Apply date filter based on config
        if START_DATE and END_DATE:
            unified_df = processor.apply_date_range_filter(unified_df, start, end)
        else:
            unified_df = processor.apply_rolling_window(unified_df, DAYS_TO_KEEP, now_jst)
    else:

        raise ValueError("No actual data found for initial load")

    # Clean up - keep only essential columns for QuickSight
    unified_df = processor.cleanup_columns(unified_df)

    # Create output folder based on date range
    output_folder = f"KPI/{START_DATE}_to_{END_DATE}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = f"{output_folder}/actual_forecast.csv"
    unified_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined data to: {output_path}")


    return {
        'rows': len(unified_df),
        "unified_df": unified_df,
        "output_folder": output_folder,
        'date_range': {
            'start': str(unified_df['datetime'].min()),
            'end': str(unified_df['datetime'].max())
        },
    }


def calculate_kpis_from_s3(unified_df, output_folder):
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

    # Initialize handlers
    metrics_calc = MetricsCalculator()

    # Load unified data from S3
    logger.info("[1/3] Loading unified data from S3...")
    
    if unified_df is None:
        raise ValueError("Unified CSV not found in S3. Run prepare_unified_data() first.")
    logger.info(f"   ✓ Loaded {len(unified_df)} rows")

    # Recalculate error metrics
    logger.info("[2/3] Calculating error metrics...")
    unified_df = metrics_calc.calculate_all_error_metrics(unified_df)
    logger.info(f"   ✓ Calculated error metrics for all {len(unified_df)} rows")
    logger.info(str(unified_df))

    # Calculate daily KPIs
    logger.info("[3/4] Calculating daily KPIs...")
    daily_kpi_df = metrics_calc.calculate_daily_kpis(unified_df)
    logger.info(f"   ✓ Calculated KPIs for {len(daily_kpi_df)} days")
    daily_kpi_df = daily_kpi_df.sort_values('date')

    # Calculate monthly KPIs
    logger.info("[4/4] Calculating monthly KPIs...")
    monthly_kpi_df = metrics_calc.calculate_monthly_kpis(unified_df)
    logger.info(f"   ✓ Calculated KPIs for {len(monthly_kpi_df)} months")

    # Save KPIs to the same output folder
    daily_kpi_path = f"{output_folder}/daily_kpis.csv"
    daily_kpi_df.to_csv(daily_kpi_path, index=False)
    logger.info(f"Saved daily KPIs to: {daily_kpi_path}")

    monthly_kpi_path = f"{output_folder}/monthly_kpis.csv"
    monthly_kpi_df.to_csv(monthly_kpi_path, index=False)
    logger.info(f"Saved monthly KPIs to: {monthly_kpi_path}")

    # Generate plots

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
        # Function 1: Prepare unified data
        data_result = prepare_unified_data()

        # Function 2: Calculate KPIs
        kpi_result = calculate_kpis_from_s3(data_result['unified_df'], data_result['output_folder'])

        # Print final summary
        logger.info("=" * 60)
        logger.info("SUCCESS - All processing complete!")
        logger.info("=" * 60)
        logger.info(f"   - Rows: {data_result['rows']}")
        logger.info(f"   - Date range: {data_result['date_range']['start']} to {data_result['date_range']['end']}")
        logger.info(f"   - Days: {kpi_result['days']}")
        logger.info(f"   - Date range: {kpi_result['date_range']['start']} to {kpi_result['date_range']['end']}")
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
