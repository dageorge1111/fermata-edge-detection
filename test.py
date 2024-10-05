import os
import json
from inference2 import input_fn, model_fn, predict_fn, output_fn
from io import BytesIO
import asyncio

# Test function to simulate running the model pipeline
async def test_locally():
    # Define your test request body
    request_body = json.dumps({
        'class_id': '602',  # Replace with an actual class_id from your Supabase table
        'uid': '89a5e09e-22b7-422c-aeda-41bd77eddb7b'  # Replace with a valid UID
    })

    # Simulate the input request
    input_data, bucket_name = input_fn(request_body)
    print("hi")
    full_input_data = (input_data, bucket_name)
    # Load the ResNet model locally
    model_dir = '../resnet-50'  # Specify the directory where 'model.pth' is stored
    model = model_fn(model_dir)

    # Run the prediction function (note the 'await')
    result = await predict_fn(full_input_data, model)
    # Print the output
    output, _ = output_fn(result)
    print(output)

# Run the test
if __name__ == "__main__":
    asyncio.run(test_locally())

