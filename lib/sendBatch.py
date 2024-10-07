import openai
import os

openai.api_key = os.getenv('OPENAI_KEY')

def process_batch(jsonl_file_path):
  try:
    with open(jsonl_file_path, 'rb') as f:
      file = openai.File.create(
        file=f,
        purpose='batch'
      )

      print(f'File uploaded: {file["id"]}')

      try:
        os.remove(jsonl_file_path)
        print(f'File {jsonl_file_path} deleted successfully.')
      except Exception as e:
        print(f'Error deleting file {jsonl_file_path}:', e)

      batch = openai.Batch.create(
        input_file_id=file['id'],
        endpoint='/v1/chat/completions',
        completion_window='24h'
      )

      print(f'Batch created: {batch["id"]}')

      return batch['id']

  except Exception as e:
    print('Error processing batch:', e)
    raise Exception(e)

if __name__ == "__main__":
  batch_id = process_batch('path_to_your_file.jsonl')
  print(f'Batch ID: {batch_id}')

