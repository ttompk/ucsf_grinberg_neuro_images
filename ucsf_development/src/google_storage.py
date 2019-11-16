# google storage functions
from google.cloud import storage, language  # library = google-cloud-storage
 # library = google-cloud-language
from io import BytesIO
import pandas as pd


class Connect_Bucket:
    
    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.bucket = self._bucket_obj()
    
    def _bucket_obj(self):
        '''connect to google storage'''
        storage_client = storage.Client(project = self.project_id)
        bucket = storage_client.get_bucket(self.bucket_name)
        return bucket
    
    def list_blobs(self):
        """Lists all the blobs in the bucket."""
        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(self.bucket_name)

        for blob in blobs:
            print(blob.name)


    def upload_blob(source_file_name, destination_blob_name):
        """ Uploads a file to the bucket. """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print('File {} uploaded to {}.'.format(
            source_file_name,
            destination_blob_name))


    def gstorage_to_pandas(self, file_to_open):
        '''open csv file from google storage as a pandas df'''
        blob = storage.blob.Blob(file_to_open, self.bucket)
        content = blob.download_as_string()
        return pd.read_csv(BytesIO(content))


