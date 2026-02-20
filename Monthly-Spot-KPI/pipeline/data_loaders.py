"""
Data Loaders Module

This module handles loading and basic transformation of data from S3:
- Actual data loading from JEPX spot data
- Forecast data loading (XGBoost, LightGBM, LSTM)
- Column standardization
- Unified CSV loading
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .config import (
    ACTUAL_DATA_PREFIX,
    OUTPUT_PREFIX,
    OUTPUT_FILENAME,
    STANDARD_COLUMNS,
    get_model_prefix,
    setup_logging
)
from .s3_operations import S3Handler

logger = setup_logging()


class DataLoader:
    """Handler for loading data from S3"""

    def __init__(self, s3_handler: S3Handler):
        self.s3 = s3_handler

    def load_actual_data(self, modified_after: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load and transform actual data from JEPX spot data

        Args:
            modified_after: Optional datetime - only load files modified after this date

        Returns:
            DataFrame with actual data
        """
        files = self.s3.list_files(ACTUAL_DATA_PREFIX, modified_after=modified_after)

        if not files:
            if modified_after:
                logger.info(f"   No files found modified after {modified_after}")
                return pd.DataFrame()
            raise ValueError(f"No CSV files found in {ACTUAL_DATA_PREFIX}")

        # Load all actual data files
        dfs = []
        for file in files:
            logger.info(f"   Reading {file}...")
            df = self.s3.read_csv(file)
            df = self._standardize_column_names(df)
            dfs.append(df)

        # Combine all files
        combined_df = pd.concat(dfs, ignore_index=True)

        # Transform to standard format
        result_df = self._transform_actual_data(combined_df)

        return result_df

    def load_forecast_data(
        self,
        model_name: str,
        modified_after: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load and combine forecast data for a specific model

        Args:
            model_name: Model name (xgboost, lightgbm, lstm, etc.)
            modified_after: Optional datetime - only load files modified after this date

        Returns:
            DataFrame with forecast data
        """
        # Get prefix from config (dynamically based on configured models)
        prefix = get_model_prefix(model_name.lower())
        if not prefix:
            raise ValueError(f"Unknown model: {model_name}. Check MODELS configuration in config.py")

        files = self.s3.list_files(prefix, modified_after=modified_after)

        if not files:
            if modified_after:
                logger.info(f"   No {model_name} files found modified after {modified_after}")
            else:
                logger.info(f"   Warning: No CSV files found in {prefix}")
            return pd.DataFrame(columns=['datetime', f'{model_name}_forecast'])

        # Load all forecast files
        dfs = []
        for file in files:
            logger.info(f"   Reading {file}...")
            df = self.s3.read_csv(file)
            dfs.append(df)

        # Combine all files
        combined_df = pd.concat(dfs, ignore_index=True)

        # Transform to standard format
        result_df = self._transform_forecast_data(combined_df, model_name)

        return result_df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names from Japanese to English

        Args:
            df: DataFrame with Japanese or English column names

        Returns:
            DataFrame with standardized English column names
        """
        # If the file already has English columns, return as-is
        if 'area_price_tohoku' in df.columns:
            logger.info("   Columns already standardized")
            return df

        # Otherwise, rename the first 15 columns
        if len(df.columns) >= 15:
            rename_map = {
                df.columns[i]: STANDARD_COLUMNS[i]
                for i in range(min(15, len(df.columns)))
            }
            df = df.rename(columns=rename_map)
            logger.info("   Standardized column names from Japanese to English")

        return df

    def _transform_actual_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform actual data to standard format

        Args:
            df: Raw actual data DataFrame

        Returns:
            Transformed DataFrame with datetime and actual_value
        """
        # Create datetime from date and time code
        # Time code: 1 = 00:00-00:30, 2 = 00:30-01:00, etc.
        df['datetime'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')

        # If date parsing failed, try other formats
        if df['datetime'].isna().any():
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

        df['datetime'] = df.apply(
            lambda row: row['datetime'] + timedelta(minutes=(row['Time code'] - 1) * 30),
            axis=1
        )

        # Create result dataframe with derived temporal features
        result_df = pd.DataFrame({
            'datetime': df['datetime'],
            'actual_value': df['area_price_tohoku'],
            'date': df['datetime'].dt.date,
            'year': df['datetime'].dt.year,
            'month': df['datetime'].dt.month,
            'day': df['datetime'].dt.day,
            'hour': df['datetime'].dt.hour,
            'day_of_week': df['datetime'].dt.day_name()
        })

        # Remove duplicates and sort
        result_df = result_df.drop_duplicates(subset=['datetime']).sort_values('datetime')

        return result_df

    def _transform_forecast_data(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Transform forecast data to standard format

        Args:
            df: Raw forecast data DataFrame
            model_name: Name of the model

        Returns:
            Transformed DataFrame with datetime and forecast values
        """
        # Convert date column to datetime
        df['datetime'] = pd.to_datetime(df['date'])

        # Create result dataframe
        result_df = pd.DataFrame({
            'datetime': df['datetime'],
            f'{model_name}_forecast': df['AreaPriceTohokuForecast']
        })

        # Remove duplicates and sort
        result_df = result_df.drop_duplicates(subset=['datetime']).sort_values('datetime')

        return result_df
