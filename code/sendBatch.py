import os
from openai import OpenAI

api_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=api_key)

def process_batch(jsonl_file_path):
  try:
    batch_input_file = client.files.create(
      file=open(jsonl_file_path, "rb"),
      purpose="batch"
    )


    try:
      os.remove(jsonl_file_path)
      print(f'File {jsonl_file_path} deleted successfully.')
    except Exception as e:
      print(f'Error deleting file {jsonl_file_path}:', e)

    batch_input_file_id = batch_input_file.id

    returned = client.batches.create(
      input_file_id=batch_input_file_id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
      metadata={
        "description": "describe edge cases"
      }
    )

    return returned.id

  except Exception as e:
    print('Error processing batch:', e)
    raise Exception(e)
