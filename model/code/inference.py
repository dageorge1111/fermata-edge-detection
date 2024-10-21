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
from descriptor import create_jsonl_from_url_list
from sendBatch import process_batch
from torchvision import models
from sklearn.ensemble import IsolationForest
from supabase import create_client
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
supabase_key = os.environ['SUPABASE_KEY']
supabase_url = os.environ['SUPABASE_URL']

def model_fn(model_dir):
    try:
        logger.info("Loading model from directory: %s", model_dir)
        resnet = models.resnet50(weights=None)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        model_path = os.path.join(model_dir, 'model.pth')
        logger.info(f"Loading model from {model_path}")
        resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        resnet.eval()
        logger.info("Model loaded successfully")
        return resnet
    except Exception as e:
        logger.error("Error loading the model: %s", str(e))
        raise

def preprocess_image(image):
    logger.info("Preprocessing image")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_image = preprocess(image).unsqueeze(0)
    logger.info("Image preprocessed")
    return processed_image

def fetch_image_from_s3(bucket, key):
    logger.info("Fetching image from S3 bucket: %s, key: %s", bucket, key)
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    img_data = response['Body'].read()
    img = Image.open(BytesIO(img_data)).convert("RGB")
    logger.info("Image fetched successfully")
    return img

def run_isolation_forest(latent_vectors_np, contamination=0.1):
    logger.info("Running Isolation Forest with contamination: %s", contamination)
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    isolation_forest.fit(latent_vectors_np)
    outlier_predictions = isolation_forest.predict(latent_vectors_np)
    outliers = np.where(outlier_predictions == -1)[0]
    logger.info("Isolation Forest completed. Found %d outliers.", len(outliers))
    return outliers, outlier_predictions

def outlier_detection(latent_vectors_np, object_ids):
    logger.info("Starting outlier detection")
    outlier_indices, predictions_isolation = run_isolation_forest(latent_vectors_np, contamination=0.1)
    outlier_ids = [object_ids[idx] for idx in outlier_indices]

    result = {
        "isolation_forest_outlier_ids": outlier_ids,
        "isolation_forest_predictions": dict(zip(object_ids, predictions_isolation.tolist()))
    }
    logger.info("Outlier detection completed")
    return result

def input_fn(request_body, content_type='application/json'):
    logger.info("Starting input_fn")
    if content_type == 'application/json':
        request = json.loads(request_body)
        class_id = request['class_id']
        uid = request['uid']
        logger.info("Received class_id: %s, uid: %s", class_id, uid)

        url = supabase_url
        key = supabase_key
        supabase = create_client(url, key)
        logger.info("Connected to Supabase")

        response = supabase.table('objects').select('*').eq('class_id', class_id).execute()
        data = response.data
        logger.info("Fetched %d objects from Supabase", len(data))

        class_response = supabase.table('classes').select('class').eq('id', class_id).execute()
        class_name = class_response.data[0]['class'] if class_response.data else None
        logger.info("Class name: %s", class_name)

        object_ids = []
        image_paths = []
        for obj in data:
            image_key = f"{uid}/{obj['set_id']}_objects/{obj['id']}"
            image_paths.append(image_key)
            object_ids.append(obj['id'])
        logger.info("Prepared image paths and object IDs")

        bucket_name = 'fermata-images'
        logger.info("Input data prepared with %d images", len(image_paths))
        return (image_paths, object_ids, class_name), bucket_name
    else:
        logger.error("Unsupported content type: %s", content_type)
        raise ValueError(f"Unsupported content type: {content_type}")

def parse_output(output_str):
    logger.info("Parsing output")
    pattern = r'\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    matches = re.findall(pattern, output_str)
    parsed = [(category.strip(), description.strip()) for category, description in matches]
    logger.info("Parsed %d output entries", len(parsed))
    return parsed

def predict_fn(input_data, model):
    logger.info("Starting predict_fn")
    s3 = boto3.client('s3')
    (image_paths, object_ids, class_name), bucket_name = input_data
    latent_vectors = []
    image_dict = {}
    outlier_urls = []

    logger.info("Processing %d images", len(image_paths))

    for idx, image_key in enumerate(image_paths):
        logger.info("Processing image %d/%d: %s", idx+1, len(image_paths), image_key)
        image = fetch_image_from_s3(bucket_name, image_key)
        image_dict[object_ids[idx]] = image
        image_preprocessed = preprocess_image(image)
        with torch.no_grad():
            latent_vector = model(image_preprocessed)
            latent_vector = latent_vector.cpu().numpy().reshape(1, -1)
            latent_vectors.append(latent_vector)

    latent_vectors_np = np.concatenate(latent_vectors, axis=0)
    logger.info("Latent vectors computed")

    result = outlier_detection(latent_vectors_np, object_ids)
    outlier_ids = result['isolation_forest_outlier_ids']
    logger.info("Found %d outliers", len(outlier_ids))
    outlier_images = {outlier_id: image_dict[outlier_id] for outlier_id in outlier_ids}

    for outlier_id in outlier_ids:
        image_key = image_paths[object_ids.index(outlier_id)]
        image_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': image_key}, ExpiresIn=3600)
        outlier_urls.append([outlier_id, image_url])
    logger.info("Generated presigned URLs for outliers")

    image_analysis = analyze_segmented_image(class_name, *outlier_urls)
    logger.info("Image analysis completed")

    parsed_tuples = parse_output(str(image_analysis))
    logger.info(str(image_analysis))
    existing = ""
    for category, description in parsed_tuples:
        logger.info("Getting tags for category: %s, description: %s", category, description)
        tags = get_tags(class_name, existing, category, description)
        existing += tags

    logger.info("Finished all tags")
    create_jsonl_from_url_list(outlier_urls, existing, "/tmp/temp.jsonl")
    logger.info("Created JSONL file 'temp.jsonl'")

    batch_id = process_batch("/tmp/temp.jsonl")
    logger.info("Batch processed with batch_id: %s", batch_id)

    return {"batch_id": batch_id}


def output_fn(prediction, accept='application/json'):
    logger.info("Starting output_fn")
    if accept == 'application/json':
        batch_id = prediction['batch_id']
        logger.info("Returning batch_id: %s", batch_id)
        return json.dumps({"batch_id": batch_id}), 'application/json'
    else:
        logger.error("Unsupported accept type: %s", accept)
        raise ValueError(f"Unsupported accept type: {accept}")

