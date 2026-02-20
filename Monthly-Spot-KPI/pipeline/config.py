"""
Configuration module for QuickSight Data Preparation Lambda

This module centralizes all configuration settings, constants, and environment variables.
"""

import os
import logging as defaultlogging
from datetime import timezone, timedelta

# AWS Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'perkins-gs')
AREA = os.environ.get('AREA', 'AreaPriceTohoku')

# S3 Path Configuration
ACTUAL_DATA_PREFIX = os.environ.get('ACTUAL_DATA_PREFIX', 'forecast/raw/jepx_spot_data/')

# Output Paths
RUN_MODE = os.environ.get('RUN_MODE', 'dev')  # 'prod' or 'dev'

# Get base output paths from environment
OUTPUT_PREFIX_BASE = os.environ.get('OUTPUT_PREFIX', 'quicksight/spot/')
TEST_OUTPUT_PREFIX_BASE = os.environ.get('TEST_OUTPUT_PREFIX', 'quicksight/spot/dev/')

# Append AREA to the output paths
if RUN_MODE == 'dev':
    OUTPUT_PREFIX = f'{TEST_OUTPUT_PREFIX_BASE}{AREA}/'
else:
    OUTPUT_PREFIX = f'{OUTPUT_PREFIX_BASE}{AREA}/'

# Output File Names
OUTPUT_FILENAME = 'latest_combined_forecast_actual.csv'
OUTPUT_DAILY_KPI_FILENAME = 'latest_daily_kpis.csv'

# Data Retention Configuration
DAYS_TO_KEEP = int(os.environ.get('DAYS_TO_KEEP', '60'))  # Default: last 51 days for dashboard

# Date Range Configuration (e.g., "2026-01-01" to "2026-02-28")
START_DATE = os.environ.get('START_DATE', "2025-11-01")  # Format: "YYYY-MM-DD"
END_DATE = os.environ.get('END_DATE', "2026-01-31")  # Format: "YYYY-MM-DD"

# Timezone Configuration
JST = timezone(timedelta(hours=9))  # Japan Standard Time (UTC+9)

# Column Names
ACTUAL_COLUMN = 'エリアプライス東北'  # Tohoku area price column

# Standard Column Mapping (Japanese -> English)
STANDARD_COLUMNS = [
    'date', 'Time code',
    'sold_bid_amount_kwh', 'buying_bid_kwh', 'contract_total_amount_kwh',
    'system_price', 'area_price_hokkaido', 'area_price_tohoku', 'area_price_tokyo',
    'area_price_central', 'area_price_hokuriku', 'area_price_kansai',
    'area_price_china', 'area_price_shikoku', 'area_price_kyushu'
]

# Encoding Options for Japanese Files
ENCODINGS = ['utf-8-sig', 'utf-8', 'shift_jis', 'cp932', 'iso-2022-jp', 'euc-jp']

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# To add or remove models, simply update this dictionary.
# The system will automatically adapt to use only the configured models.
#
# To DISABLE a model: Comment out or remove its entry
# To ADD a model: Add a new entry with model_key, display_name, and s3_prefix
#
# Example to disable LSTM:
#   Comment out the 'lstm' entry below
#
# Example to add a new model:
#   'prophet': {
#       'name': 'Prophet',
#       'prefix': f'forecast/spot/{AREA}/prophet/forecast_data/'
#   }
# ============================================================================

MODELS = {
    'xgboost': {
        'name': 'XGBoost',
        'prefix': f'forecast/spot/{AREA}/boosting/layer_3_forecast_data/XGBoost/'
    },
    'lightgbm': {
        'name': 'LightGBM',
        'prefix': f'forecast/spot/{AREA}/boosting/layer_3_forecast_data/LightGBM/'
    },
    "gradient_boosting": {
        'name': 'Gradient Boosting',
        'prefix': f'forecast/spot/{AREA}/boosting/layer_3_forecast_data/GradientBoosting/'
    },
    'catboost': {
        'name': 'CatBoost',
        'prefix': f'forecast/spot/{AREA}/boosting/layer_3_forecast_data/CatBoost/'
    },
    'adaboost': {
        'name': 'AdaBoost',
        'prefix': f'forecast/spot/{AREA}/boosting/layer_3_forecast_data/AdaBoost/'
    },
    'lstm': {
        'name': 'LSTM',
        'prefix': f'forecast/spot/{AREA}/lstm/layer_2_forecast_data/'
    },
}

featured_data = {
     'lstm': {
        'name': 'LSTM',
        'prefix': f'forecast/spot/{AREA}/lstm/layer_1_forecast_data/'
    },
}
 

# Dynamically generate essential columns based on active models
def get_essential_columns():
    """Generate essential columns list based on active models"""
    columns = ['datetime', 'actual_value']
    for model_key in MODELS.keys():
        columns.append(f'{model_key}_forecast')
    return columns

ESSENTIAL_COLUMNS = get_essential_columns()

# Helper function to get model names
def get_model_names():
    """Get list of model keys (xgboost, lightgbm, lstm, etc.)"""
    return list(MODELS.keys())

def get_model_display_name(model_key):
    """Get display name for a model"""
    return MODELS.get(model_key, {}).get('name', model_key.upper())

def get_model_prefix(model_key):
    """Get S3 prefix for a model"""
    return MODELS.get(model_key, {}).get('prefix', '')


def setup_logging():
    """
    Configure logging for the entire application

    This should be called once at the start of the Lambda function.
    After this, each module can simply use:
        logger = logging.getLogger(__name__)

    The logger will automatically use this configuration.
    """

    # Create named logger
    logging = defaultlogging.getLogger("my_application")

    # Set level from env (default to INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.setLevel(getattr(defaultlogging, log_level, defaultlogging.INFO))

    # Remove existing handlers if any (avoid duplicate logs in Lambda cold start)
    if logging.hasHandlers():
        logging.handlers.clear()

    # Stream handler (console)
    stream_handler = defaultlogging.StreamHandler()
    stream_formatter = defaultlogging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    logging.addHandler(stream_handler)

    # Optional: Reduce noise from other libraries
    defaultlogging.getLogger().setLevel(defaultlogging.WARNING)

    return logging
