"""
S3 Operations Module

This module handles all AWS S3 interactions including:
- File existence checks
- Listing files with filtering
- Reading CSV files with encoding detection
- Writing CSV files to S3
"""

import boto3
import pandas as pd
from io import StringIO
from typing import List, Optional
from datetime import datetime

from .config import S3_BUCKET, ENCODINGS, setup_logging

logger = setup_logging()


class S3Handler:
    """Handler for S3 operations"""

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket = S3_BUCKET

    def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3

        Args:
            key: S3 object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

    def list_files(self, prefix: str, modified_after: Optional[datetime] = None) -> List[str]:
        """
        List all CSV files in S3 prefix, optionally filtered by last modified date

        Args:
            prefix: S3 prefix to search
            modified_after: Optional datetime object (timezone-aware) -
                          only return files modified after this date

        Returns:
            List of S3 keys matching criteria
        """
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.csv'):
                        # Filter by last modified date if specified
                        if modified_after:
                            obj_modified = obj['LastModified']
                            if obj_modified >= modified_after:
                                files.append(obj['Key'])
                        else:
                            files.append(obj['Key'])

        return files

    def read_csv(self, key: str) -> pd.DataFrame:
        """
        Read CSV file from S3 with automatic encoding detection

        Args:
            key: S3 object key

        Returns:
            DataFrame with CSV contents

        Raises:
            UnicodeDecodeError: If file cannot be decoded with any encoding
        """
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        raw_content = response['Body'].read()

        # Try multiple encodings for Japanese files
        for encoding in ENCODINGS:
            try:
                content = raw_content.decode(encoding)
                df = pd.read_csv(StringIO(content))
                logger.info(f"   Successfully decoded with {encoding}")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        # If all encodings fail, raise error
        raise UnicodeDecodeError(f"Unable to decode file {key} with any encoding")

    def write_csv(self, df: pd.DataFrame, key: str) -> None:
        """
        Write DataFrame to S3 as CSV

        Args:
            df: DataFrame to save
            key: S3 object key
        """
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )

        logger.info(f"   ✓ Saved {key} ({len(df)} rows)")

    def get_file_path(self, prefix: str, filename: str) -> str:
        """
        Construct full S3 path

        Args:
            prefix: S3 prefix
            filename: File name

        Returns:
            Full S3 path
        """
        return f"{prefix}{filename}"

    def get_s3_uri(self, key: str) -> str:
        """
        Get full S3 URI for a key

        Args:
            key: S3 object key

        Returns:
            S3 URI (s3://bucket/key)
        """
        return f"s3://{self.bucket}/{key}"
