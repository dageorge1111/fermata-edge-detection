import os
import json

# Convert an array of JSON objects into JSONL format
def convert_to_jsonl(data):
  return "\n".join([json.dumps(item) for item in data])

# Function to create a JSONL file from url_list and descriptor_string
def create_jsonl_from_url_list(url_list, descriptor_string, file_name):
  if not isinstance(url_list, list) or len(url_list) == 0:
    raise ValueError('Invalid url_list. Expected a non-empty list.')
    
  if not isinstance(descriptor_string, str) or not descriptor_string.strip():
    raise ValueError('Invalid descriptor_string. Expected a non-empty string.')
    
  if not isinstance(file_name, str) or not file_name.strip():
    raise ValueError('Invalid file name. Expected a non-empty string.')

    # Ensure the file name has a .jsonl extension
  if not file_name.endswith('.jsonl'):
    file_name += '.jsonl'

    # Generate the data for each [id, image_url] in the url_list
  jsonl_data = []
  for entry in url_list:
    if len(entry) != 2:
      raise ValueError(f"Invalid entry format: {entry}. Expected [id, image_url].")
        
      id_, image_url = entry
      jsonl_data.append({
        "custom_id": str(id_),  # Use id as custom_id
        "method": "POST",  # The HTTP method to be used for the request
        "url": "/v1/chat/completions",  # The OpenAI API endpoint
        "body": {
          "model": "gpt-4o-mini",  # Specify the model
          "messages": [
            {
              "role": "system",
              "content": [
                {
                  "type": "text",
                  "text": ("You are an image analyzer who is going to look at the user's image "
                  "and classify it with the comma-separated list of tuples passed in by the user. "
                  "Each tuple contains an id and a list of classifications for the object in the image. "
                  "Pick one classification from each tuple and return a comma separated list in the "
                  "following format: (id1, classification1), (id2, classification2)...")
                }
              ]
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": (f"Look at the image. Focus exclusively on the main object in this image and nothing else. "
                           f"Review these tuples: {descriptor_string}.")
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": image_url  # The image URL for this entry
                  }
                }
              ]
            }
          ],
          "temperature": 0.3  # Temperature setting for the completion
        }
      })

    # Convert the JSON data to JSONL format
  jsonl_content = convert_to_jsonl(jsonl_data)

    # Define the file path with the provided file name
  file_path = os.path.join(os.getcwd(), file_name)

    # Write the JSONL data to a file
  with open(file_path, 'w') as file:
    file.write(jsonl_content)

  print(f"JSONL file created at: {file_path}")
  return file_path
