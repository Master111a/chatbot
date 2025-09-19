#!/usr/bin/env python3
import os
from minio import Minio
from minio.error import S3Error

def create_bucket(client, bucket_name):
    """Create bucket if it doesn't exist"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' is created")
        else:
            print(f"Bucket '{bucket_name}' is already exists")
    except S3Error as err:
        print(f"Error creating bucket '{bucket_name}': {err}")

def main():
    minio_host = os.environ.get('MINIO_HOST', 'minio')
    minio_port = os.environ.get('MINIO_PORT', '9000')
    minio_access_key = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
    minio_secret_key = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
    
    try:
        client = Minio(
            f"{minio_host}:{minio_port}",
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        print("Successfully connected to MinIO Server")

        buckets = [
            "newwave-documents",
        ]
        
        for bucket in buckets:
            create_bucket(client, bucket)
            
        print("Completed initializing MinIO buckets")
        
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        return

if __name__ == "__main__":
    main() 