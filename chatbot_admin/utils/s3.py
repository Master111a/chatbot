import boto3
from botocore.config import Config
from django.conf import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class MinioService:
    def __init__(self):
        self.s3 = self._get_s3_client()
        self.bucket = self.s3.Bucket(settings.MINIO_BUCKET_NAME)

    def _get_s3_client(self):
        """Initialize S3 client for MinIO"""
        return boto3.resource(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',
        )

    def download_file(self, object_name):
        try:
            filename = object_name.split('/')[-1]

            url = self.s3.meta.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.MINIO_BUCKET_NAME,
                    'Key': object_name,
                    'ResponseContentDisposition': f'attachment; filename="{filename}"'
                },
                ExpiresIn=3600
            )
            return url
        except Exception as e:
            logger.error(f"Error generating download URL: {str(e)}")
            return None