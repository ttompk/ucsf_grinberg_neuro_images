# google storage functions
from google.cloud import storage, language  
 # library = google-cloud-storage   # must be v1.17 or greater
 # library = google-cloud-language
from io import BytesIO
import pandas as pd


class Connect_Bucket:
    
    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = None
        if storage.__version__ >= '1.17.0':
            self.bucket = self._bucket_obj()
        else: 
            print("Error: Please upgrade 'google-cloud-storage' package version to >= 1.17.0 ")
    
    def _bucket_obj(self):
        '''connect to google storage'''
        self.storage_client = storage.Client(project = self.project_id)
        bucket = self.storage_client.get_bucket(self.bucket_name)
        return bucket
    
    def list_blobs(self):
        """Lists all the blobs in the bucket."""
        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(self.bucket_name)

        for blob in blobs:
            print(blob.name)

    def get_file(self, file_path):
        '''grab a blob from a bucket'''
        print(self.bucket)
        blob = storage.blob.Blob(file_path, self.bucket)
        fileName = blob.name.split('/')[-1]
        file_obj = BytesIO()
        #blob.download_to_file(file_obj)
        #return blob.download_to_file(file_obj, client=self.storage_client)
        return blob.download_to_filename(filename)  
            
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


