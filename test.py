import boto3
import json
import time
from urllib.parse import urlparse

runtime_client = boto3.client('sagemaker-runtime', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')

import boto3

endpoint_name = 'fermata-edge-detection-endpoint'

input_data = {
    'class_id': '602',
    'uid': '89a5e09e-22b7-422c-aeda-41bd77eddb7b'
}

payload = json.dumps(input_data)

response = runtime_client.invoke_endpoint_async(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Accept='application/json',
    InputLocation='s3://fermata-edge-detection/input.json'
)

output_location = response['OutputLocation']
print(f"Output will be stored at: {output_location}")

parsed_url = urlparse(output_location)
bucket_name = parsed_url.netloc
object_key = parsed_url.path.lstrip('/')

while True:
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        result = response['Body'].read().decode('utf-8')
        print("Inference result:")
        print(result)
        break
    except s3_client.exceptions.NoSuchKey:
        print("Result not ready yet. Waiting for 5 seconds...")
        time.sleep(5)
