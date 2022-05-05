import boto3
import time

def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    # Download a 20MB file from S3
    with open('/tmp/20MB', 'wb') as data:
        s3_client.download_fileobj('serverlessappperfopt-network-intensive-source-bucket', '20MB', data)
    return {
        'statusCode': 200,
        'body': {"name":"f4","input":event}
    }
