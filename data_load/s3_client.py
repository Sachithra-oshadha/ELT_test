import boto3
import os
import re

class S3Client:
    def __init__(self, s3_config, bucket_name, prefix, logger):
        self.s3_config = s3_config
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.logger = logger  # Use passed logger
        self.client = None

    def connect(self):
        try:
            self.client = boto3.client(
                's3',
                aws_access_key_id=self.s3_config['aws_access_key_id'],
                aws_secret_access_key=self.s3_config['aws_secret_access_key'],
                region_name=self.s3_config['region_name']
            )
            self.client.list_buckets()
            self.logger.info("Successfully connected to S3")
        except Exception as e:
            self.logger.error(f"Failed to connect to S3: {e}")
            raise

    def list_files(self):
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if re.search(r'\.(csv|xlsx|xls)$', key, re.IGNORECASE):
                            files.append(key)
            self.logger.info(f"Found {len(files)} Excel/CSV files in S3 bucket")
            return files
        except Exception as e:
            self.logger.error(f"Failed to list S3 files: {e}")
            raise

    def download_file(self, s3_key, temp_dir):
        try:
            local_path = os.path.join(temp_dir, os.path.basename(s3_key))
            self.client.download_file(self.bucket_name, s3_key, local_path)
            self.logger.info(f"Downloaded file from S3: {s3_key} to {local_path}")
            return local_path
        except Exception as e:
            self.logger.error(f"Failed to download S3 file {s3_key}: {e}")
            raise