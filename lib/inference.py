import os
import json
import torch
import torchvision.transforms as transforms
import re
from PIL import Image
import boto3
from io import BytesIO
import torch.nn as nn
import numpy as np
from locator import analyze_segmented_image
from locator import get_tags 
from torchvision import models
from sklearn.ensemble import IsolationForest
from supabase import create_client
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
supabase_key = os.getenv('SUPABASE_KEY')
supabase_url = os.getenv('SUPABASE_URL')

# Function to load the ResNet model
def model_fn(model_dir):
    resnet = models.resnet50(weights=None)
    resnet = nn.Sequential(*list(resnet.children())[:-1]) 
    model_path = os.path.join(model_dir, 'model.pth')
    resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
    resnet.eval()
    return resnet

# Function to preprocess images (resize, normalize)
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize without cropping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to fetch images from S3
def fetch_image_from_s3(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    img_data = response['Body'].read()
    img = Image.open(BytesIO(img_data)).convert("RGB")  # Ensure it's in RGB format
    return img  # Return PIL Image without preprocessing

# Isolation Forest for Outlier Detection
def run_isolation_forest(latent_vectors_np, contamination=0.1):
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    isolation_forest.fit(latent_vectors_np)

    # Predict outliers (-1 for outliers, 1 for normal points)
    outlier_predictions = isolation_forest.predict(latent_vectors_np)
    outliers = np.where(outlier_predictions == -1)[0]

    return outliers, outlier_predictions

# Function to perform outlier detection using Isolation Forest
def outlier_detection(latent_vectors_np, object_ids):
    # Perform Isolation Forest
    outlier_indices, predictions_isolation = run_isolation_forest(latent_vectors_np, contamination=0.1)

    # Map outlier indices to object IDs
    outlier_ids = [object_ids[idx] for idx in outlier_indices]

    result = {
        "isolation_forest_outlier_ids": outlier_ids,  # List of object IDs identified as outliers
        "isolation_forest_predictions": dict(zip(object_ids, predictions_isolation.tolist()))
    }
    return result

# Function to process incoming request and prepare input data
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        # Parse the request body
        request = json.loads(request_body)
        class_id = request['class_id']
        uid = request['uid']
        # Initialize Supabase client
        url = supabase_url
        key =  supabase_key
        supabase = create_client(url, key)
        response = supabase.table('objects').select('*').eq('class_id', class_id).execute()
        data = response.data

        class_response = supabase.table('classes').select('class').eq('id', class_id).execute()
        class_name = class_response.data[0]['class'] if class_response.data else None

        object_ids = []
        image_paths = []
        for obj in data:
            image_key = f"{uid}/{obj['set_id']}_objects/{obj['id']}"
            image_paths.append(image_key)
            object_ids.append(obj['id'])

        bucket_name = 'fermata-images'

        return (image_paths, object_ids, class_name), bucket_name
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def parse_output(output_str):
    pattern = r'\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    matches = re.findall(pattern, output_str)
    return [(category.strip(), description.strip()) for category, description in matches]

async def predict_fn(input_data, model):
    print(input_data)
    (image_paths, object_ids, class_name), bucket_name = input_data
    latent_vectors = []
    image_dict = {}  # To store images for later use
    outlier_urls = []  # Store outlier URLs

    for idx, image_key in enumerate(image_paths):
        image = fetch_image_from_s3(bucket_name, image_key)
        image_dict[object_ids[idx]] = image  # Store images with their IDs
        image_preprocessed = preprocess_image(image)
        with torch.no_grad():
            latent_vector = model(image_preprocessed)  # Pass image through the model
            latent_vector = latent_vector.cpu().numpy().reshape(1, -1) 
            latent_vectors.append(latent_vector)

    latent_vectors_np = np.concatenate(latent_vectors, axis=0)  # Create a NumPy array from latent vectors

    result = outlier_detection(latent_vectors_np, object_ids)
    outlier_ids = result['isolation_forest_outlier_ids']
    outlier_images = {outlier_id: image_dict[outlier_id] for outlier_id in outlier_ids}

    # Generate outlier image URLs for further processing
    for outlier_id in outlier_ids:
        image_key = image_paths[object_ids.index(outlier_id)]
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"  # Construct S3 image URL
        outlier_urls.append(image_url)
#get_tags(image_class, existing, descriptor, definition):
    # Call the analyze_segmented_image function from locator.py
    image_analysis = await analyze_segmented_image(class_name, *outlier_urls)

    parsed_tuples = parse_output(str(image_analysis))
    existing = ""
    for i in parsed_tuples:
        tags = await get_tags(class_name, existing, i[0], i[1])
        existing += tags
        print(tags)
    return {"outlier_urls": outlier_urls, "class_name": class_name, "image_analysis": image_analysis}

def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        # Use 'outlier_urls' instead of 'outlier_images' and return the class name as well
        outlier_urls = prediction['outlier_urls']
        class_name = prediction['class_name']
        return json.dumps({"outlier_urls": outlier_urls, "class_name": class_name}), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
