import os
import re
import shutil
import boto3
from botocore.exceptions import BotoCoreError, ClientError


class FileClient:
    """
    Unified file client supporting:
    - AWS S3 (preferred)
    - Local filesystem (fallback)

    Public API:
        - connect()
        - list_files()
        - download_file(file_key, temp_dir)
    """

    def __init__(
        self,
        logger,
        s3_config=None,
        bucket_name=None,
        prefix=None,
        local_input_dir=None
    ):
        self.logger = logger

        # S3 config (optional)
        self.s3_config = s3_config
        self.bucket_name = bucket_name
        self.prefix = prefix or ""

        # Local config
        self.local_input_dir = local_input_dir

        self.s3_client = None
        self.mode = None  # "s3" or "local"

    # --------------------------------------------------
    # Connection handling
    # --------------------------------------------------
    def connect(self):
        """
        Try to connect to S3 first.
        If it fails, fall back to local filesystem.
        """
        if self._try_s3():
            self.mode = "s3"
            self.logger.info("Storage mode selected: S3")
            return

        self._use_local()

    def _try_s3(self):
        """
        Attempt to establish an S3 connection.
        Returns True if successful, False otherwise.
        """
        if not self.s3_config or not self.bucket_name:
            self.logger.warning("S3 config not provided. Skipping S3.")
            return False

        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.s3_config.get("aws_access_key_id"),
                aws_secret_access_key=self.s3_config.get("aws_secret_access_key"),
                region_name=self.s3_config.get("region_name")
            )

            # Test connectivity
            self.s3_client.list_buckets()
            return True

        except (BotoCoreError, ClientError, Exception) as e:
            self.logger.warning(f"S3 unavailable: {e}")
            return False

    def _use_local(self):
        """
        Activate local filesystem mode.
        """
        if not self.local_input_dir:
            raise RuntimeError(
                "Local storage selected but LOCAL_INPUT_DIR is not configured"
            )

        if not os.path.isdir(self.local_input_dir):
            raise RuntimeError(
                f"Local input directory does not exist: {self.local_input_dir}"
            )

        self.mode = "local"
        self.logger.info(f"Storage mode selected: LOCAL ({self.local_input_dir})")

    # --------------------------------------------------
    # File operations
    # --------------------------------------------------
    def list_files(self):
        """
        List available CSV / Excel files.
        Returns a list of file identifiers:
        - S3: object keys
        - Local: absolute file paths
        """
        if self.mode == "s3":
            return self._list_s3_files()
        return self._list_local_files()

    def _list_s3_files(self):
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=self.prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if re.search(r"\.(csv|xlsx|xls)$", key, re.IGNORECASE):
                    files.append(key)

        self.logger.info(f"Found {len(files)} files in S3")
        return files

    def _list_local_files(self):
        files = []

        for root, _, filenames in os.walk(self.local_input_dir):
            for name in filenames:
                if re.search(r"\.(csv|xlsx|xls)$", name, re.IGNORECASE):
                    files.append(os.path.join(root, name))

        self.logger.info(f"Found {len(files)} files in local storage")
        return files

    def download_file(self, file_key, temp_dir):
        """
        Copies file into temp_dir and returns local path.
        """
        if self.mode == "s3":
            return self._download_from_s3(file_key, temp_dir)
        return self._copy_local_file(file_key, temp_dir)

    def _download_from_s3(self, s3_key, temp_dir):
        local_path = os.path.join(temp_dir, os.path.basename(s3_key))
        self.s3_client.download_file(
            self.bucket_name,
            s3_key,
            local_path
        )
        self.logger.info(f"Downloaded from S3: {s3_key}")
        return local_path

    def _copy_local_file(self, file_path, temp_dir):
        local_path = os.path.join(temp_dir, os.path.basename(file_path))
        shutil.copy2(file_path, local_path)
        self.logger.info(f"Copied local file: {file_path}")
        return local_path