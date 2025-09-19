import requests

from django.conf import settings

class DocumentAPI:
    BASE_URL = getattr(settings, 'FASTAPI_BASE_URL')
    PUBLIC_URL = getattr(settings, 'PUBLIC_URL')

    @classmethod
    def upload_document(cls, header, metadata, files):
        url = f"{cls.BASE_URL}/documents/upload"
        time_out = 600
        try:
            response = requests.post(url, headers=header, data=metadata, files=files, timeout=time_out)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to upload document: {e}")

    @classmethod
    def batch_upload_documents(cls, header, metadata, files):
        url = f"{cls.BASE_URL}/documents/batch-upload"
        time_out = 600
        try:
            response = requests.post(url, headers=header, data=metadata, files=files, timeout=time_out)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to batch upload documents: {e}")

    @classmethod
    def delete_document(cls, header, document_id):
        url = f"{cls.BASE_URL}/documents/{str(document_id)}"

        try:
            response = requests.delete(url, headers=header, timeout=600)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to delete document: {e}")

    @classmethod
    def download_document(cls, document_id):
        url = f"{cls.PUBLIC_URL}/documents/download/{str(document_id)}"
        try:
            return url
        except requests.RequestException as e:
                raise Exception(f"Failed to download document: {e}")

    @classmethod
    def get_document_details(cls, header, document_id):
        url = f"{cls.BASE_URL}/documents/{document_id}"

        try:
            response = requests.get(url, headers=header, timeout=600)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to retrieve document: {e}")
