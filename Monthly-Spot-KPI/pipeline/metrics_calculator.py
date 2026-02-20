"""
Metrics Calculator Module

This module handles calculation of error metrics and KPIs:
- Individual error metrics (MAE, RMSE, SMAPE)
- Daily aggregated KPIs
- Model-specific calculations
"""

import pandas as pd
from typing import Dict, List

from .config import get_model_names


class MetricsCalculator:
    """Handler for calculating error metrics and KPIs"""

    @staticmethod
    def calculate_error_metrics(
        df: pd.DataFrame,
        model_name: str
    ) -> pd.DataFrame:
        """
        Calculate error metrics for a specific model

        Args:
            df: DataFrame with actual_value and model_forecast columns
            model_name: Name of the model (xgboost, lightgbm, lstm)

        Returns:
            DataFrame with error metrics added
        """
        forecast_col = f'{model_name}_forecast'

        if forecast_col not in df.columns or not df[forecast_col].notna().any():
            return df

        # Calculate error (actual - forecast)
        df[f'{model_name}_error'] = df['actual_value'] - df[forecast_col]

        # Calculate absolute error
        df[f'{model_name}_abs_error'] = df[f'{model_name}_error'].abs()

        # Calculate squared error
        df[f'{model_name}_squared_error'] = df[f'{model_name}_error'] ** 2

        # Calculate SMAPE: 100 * |actual - forecast| / ((|actual| + |forecast|) / 2)
        denominator = df['actual_value'].abs() + df[forecast_col].abs()
        df[f'{model_name}_smape'] = (
            200 * df[f'{model_name}_abs_error'] / denominator
        ).where(denominator != 0, pd.NA)

        return df

    @staticmethod
    def calculate_all_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate error metrics for all configured models

        Args:
            df: DataFrame with actual_value and forecast columns

        Returns:
            DataFrame with all error metrics added
        """
        for model_name in get_model_names():
            df = MetricsCalculator.calculate_error_metrics(df, model_name)

        return df

    @staticmethod
    def calculate_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily aggregated KPIs for all models

        Args:
            df: DataFrame with datetime, actual_value, forecasts, and error metrics

        Returns:
            DataFrame with daily KPIs (MAE, RMSE, SMAPE) for each model
        """
        # Derive date from datetime for grouping
        df = df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        daily_kpis = []

        # Group by date and calculate KPIs
        for date, group in df.groupby('date'):
            kpi_row = {'date': date}

            # Calculate KPIs for each configured model
            for model_name in get_model_names():
                model_kpis = MetricsCalculator._calculate_model_daily_kpi(
                    group,
                    model_name
                )
                kpi_row.update(model_kpis)

            daily_kpis.append(kpi_row)

        # Create DataFrame and sort by date
        daily_kpi_df = pd.DataFrame(daily_kpis)
        daily_kpi_df = daily_kpi_df.sort_values('date')

        return daily_kpi_df

    @staticmethod
    def calculate_monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly aggregated KPIs for all models

        Args:
            df: DataFrame with datetime, actual_value, forecasts, and error metrics

        Returns:
            DataFrame with monthly KPIs (MAE, RMSE, SMAPE) for each model
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year_month'] = df['datetime'].dt.to_period('M')

        monthly_kpis = []

        for year_month, group in df.groupby('year_month'):
            kpi_row = {'month': str(year_month)}

            for model_name in get_model_names():
                model_kpis = MetricsCalculator._calculate_model_monthly_kpi(
                    group,
                    model_name
                )
                kpi_row.update(model_kpis)

            monthly_kpis.append(kpi_row)

        monthly_kpi_df = pd.DataFrame(monthly_kpis)
        monthly_kpi_df = monthly_kpi_df.sort_values('month')

        return monthly_kpi_df

    @staticmethod
    def _calculate_model_monthly_kpi(
        group: pd.DataFrame,
        model_name: str
    ) -> dict:
        """
        Calculate monthly KPIs for a specific model

        Args:
            group: DataFrame group for a specific month
            model_name: Name of the model

        Returns:
            Dictionary with model KPIs
        """
        kpis = {}
        forecast_col = f'{model_name}_forecast'

        if forecast_col not in group.columns:
            return kpis

        valid_data = group.dropna(subset=[forecast_col, 'actual_value'])

        if len(valid_data) > 0:
            kpis[f'{model_name}_mae'] = valid_data[f'{model_name}_abs_error'].mean()
            kpis[f'{model_name}_rmse'] = (
                valid_data[f'{model_name}_squared_error'].mean()
            ) ** 0.5
            kpis[f'{model_name}_smape'] = valid_data[f'{model_name}_smape'].mean()

        return kpis

    @staticmethod
    def _calculate_model_daily_kpi(
        group: pd.DataFrame,
        model_name: str
    ) -> Dict[str, float]:
        """
        Calculate daily KPIs for a specific model

        Args:
            group: DataFrame group for a specific date
            model_name: Name of the model

        Returns:
            Dictionary with model KPIs
        """
        kpis = {}
        forecast_col = f'{model_name}_forecast'

        if forecast_col not in group.columns:
            return kpis

        # Filter to valid (non-null) forecasts
        valid_data = group.dropna(subset=[forecast_col, 'actual_value'])

        if len(valid_data) > 0:
            # MAE (Mean Absolute Error)
            kpis[f'{model_name}_mae'] = valid_data[f'{model_name}_abs_error'].mean()

            # RMSE (Root Mean Squared Error)
            kpis[f'{model_name}_rmse'] = (
                valid_data[f'{model_name}_squared_error'].mean()
            ) ** 0.5

            # SMAPE (Symmetric Mean Absolute Percentage Error)
            kpis[f'{model_name}_smape'] = valid_data[f'{model_name}_smape'].mean()

        return kpis

    @staticmethod
    def get_kpi_summary(daily_kpi_df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics for KPIs

        Args:
            daily_kpi_df: DataFrame with daily KPIs

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_days': len(daily_kpi_df),
            'date_range': {
                'start': str(daily_kpi_df['date'].min()),
                'end': str(daily_kpi_df['date'].max())
            },
            'models': {}
        }

        # Calculate average KPIs for each configured model
        for model_name in get_model_names():
            mae_col = f'{model_name}_mae'
            rmse_col = f'{model_name}_rmse'
            smape_col = f'{model_name}_smape'

            if mae_col in daily_kpi_df.columns:
                summary['models'][model_name] = {
                    'avg_mae': float(daily_kpi_df[mae_col].mean()),
                    'avg_rmse': float(daily_kpi_df[rmse_col].mean()),
                    'avg_smape': float(daily_kpi_df[smape_col].mean())
                }

        return summary
