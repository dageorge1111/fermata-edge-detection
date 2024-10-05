# Use the official AWS PyTorch base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.0-cpu-py36-ubuntu18.04

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install additional dependencies
RUN pip install scikit-learn boto3 requests pytorch-msssim

# Copy the inference script into the Docker container
COPY inference.py /opt/ml/model/code/inference.py

# Set the entry point for SageMaker to run inference.py
ENV SAGEMAKER_PROGRAM inference.py
