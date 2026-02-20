"""
Data Processors Module

This module handles data processing operations:
- Merging actual and forecast data
- Applying rolling windows
- Deduplication
- Data filtering
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .config import DAYS_TO_KEEP, START_DATE, END_DATE, JST, ESSENTIAL_COLUMNS, setup_logging

logger = setup_logging()


class DataProcessor:
    """Handler for data processing operations"""

    @staticmethod
    def merge_datasets(
        actual_df: pd.DataFrame,
        forecast_dfs: dict
    ) -> pd.DataFrame:
        """
        Merge actual data with all forecast datasets on datetime

        Uses OUTER join to preserve:
        - All actual data (even if no forecast exists yet)
        - All forecast data (even if actual data doesn't exist yet for future dates)

        Args:
            actual_df: Actual data DataFrame
            forecast_dfs: Dictionary of {model_name: forecast_df} for all models

        Returns:
            Merged DataFrame with all forecasts
        """
        # Start with actual data
        merged_df = actual_df.copy()

        # Dynamically merge all forecast models using OUTER join
        for model_name, forecast_df in forecast_dfs.items():
            forecast_col = f'{model_name}_forecast'
            logger.info(f"   Merging {model_name} forecasts into actual data...")
            logger.info(forecast_df)

            if not forecast_df.empty:
                # Use OUTER join to keep all dates from both actual and forecast
                merged_df = merged_df.merge(forecast_df, on='datetime', how='outer')
                logger.info(f"   ✓ Merged {model_name} forecasts (outer join)")
                logger.info(merged_df)
            else:
                merged_df[forecast_col] = pd.NA
                logger.info(f"   ⚠ No {model_name} forecasts to merge, added empty column")

        # Sort by datetime to maintain chronological order
        merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

        return merged_df

    @staticmethod
    def apply_rolling_window(
        df: pd.DataFrame,
        days_to_keep: int = DAYS_TO_KEEP,
        reference_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Apply rolling window to keep only recent data

        Args:
            df: DataFrame to filter
            days_to_keep: Number of days to keep
            reference_time: Reference time for cutoff (defaults to now in JST)

        Returns:
            Filtered DataFrame
        """
        if days_to_keep <= 0:
            return df

        if reference_time is None:
            reference_time = datetime.now(JST)

        # Remove timezone for comparison
        cutoff_date = reference_time.replace(tzinfo=None) - timedelta(days=days_to_keep)
        before_filter = len(df)
        df_filtered = df[df['datetime'] >= cutoff_date]

        rows_removed = before_filter - len(df_filtered)
        logger.info(f"   ✓ Applied rolling window: kept last {days_to_keep} days ({cutoff_date.date()} onwards)")
        if rows_removed > 0:
            logger.info(f"   ✓ Removed {rows_removed} rows outside the window")

        return df_filtered

    @staticmethod
    def apply_date_range_filter(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Filter data to a specific date range

        Args:
            df: DataFrame to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        before_filter = len(df)

        # Remove timezone for comparison if needed
        start = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        end = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date

        df_filtered = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

        rows_removed = before_filter - len(df_filtered)
        logger.info(f"   ✓ Applied date range filter: {start.date()} to {end.date()}")
        logger.info(f"   ✓ Kept {len(df_filtered)} rows, removed {rows_removed} rows")

        return df_filtered

    @staticmethod
    def cleanup_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only essential columns for QuickSight

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with only essential columns
        """
        columns_to_keep = [col for col in ESSENTIAL_COLUMNS if col in df.columns]
        df_clean = df[columns_to_keep]
        logger.info(f"   ✓ Cleaned unified dataset to {len(columns_to_keep)} columns (removed date/time parts)")
        return df_clean

