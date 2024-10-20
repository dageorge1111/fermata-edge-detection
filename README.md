# Edge Detection for Fermata
This code is inspired by [Identifying and Generating Edge Cases](https://dl.acm.org/doi/pdf/10.1145/3665451.3665529) and [Deep Embedded K-Means Clustering](https://ieeexplore.ieee.org/document/9680003)

This will analyze image datasets stored in cloud storage and pull out latent features. It will then use Isolated Forest to pick outliers from a lower dimensional latent space. Then it will use GPT-4o-mini to pull out standardized descriptions for each outlier.

## Deep Embedding:
We used a pre-trained Resnet-50 model to pull out valuable latent features from 224 x 224 RGB iamges. We cut off the last layer of the CNN to get the latent features after pooling without the actual classification. This can be found in lib/inference.py.
![Resnet Image](RESNET-50.png)

## Usage:
These steps outline how to create a sagemaker endpoint for the edge case analysis.
1. **Navigate to the model directory:**
   ```bash
   cd model
   ```
2. **Create a tarball file for the model:**
   ```bash
   tar -czvf model.tar.gz model.pth code/
   ```
3. **Migrate model to s3 bucket:**
   ```bash
   aws s3 cp model.tar.gz s3://<your-bucket-name>/
   ```
4. **Run deploy.py**
