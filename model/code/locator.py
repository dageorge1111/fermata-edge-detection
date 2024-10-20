from flask import Flask, request, jsonify
from supabase import create_client, Client
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
supabase_key = os.environ['SUPABASE_KEY']
supabase_url = os.environ['SUPABASE_URL']

# Set up Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

openai_key = os.environ['OPENAI_KEY']
client = OpenAI(api_key=openai_key)

def extract_tuples(input_string):
    inner_content = input_string.strip()[1:-1]
    # Updated pattern to match inner tuples while ignoring the outermost parentheses
    pattern = r'\(([^,]+),\s*([^)]+)\)'
 
    # Find all tuples in the string using regex
    matches = re.findall(pattern, inner_content)
    
    # Return the matches as a list of tuples
    result = [(match[0].strip(), match[1].strip()) for match in matches]
    
    return result
def analyze_segmented_image(image_class, *image_urls):
    """Analyze segmented images using OpenAI synchronously"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a system that takes in example images of a type of object. You will first determine what object is in all the images, then generate generic category types for the type of object. For example, if the object were humans, you would generate (skin color, shirt color, pants color, hair color, orientation, moving). For each of these category types, create a short description of what each category means. For example, if the object were humans, you would generate ((skin color, the color of the human’s skin), (shirt color, the color of the human’s shirt), (pants color, the color of the human’s pants), (hair color, the color of the human’s hair), (orientation, a direction that the human is facing), (moving, the direction that the human is moving in). I only want you to output these generic types that are visibly discernible and their short descriptions. Output all types in the following format ((category1, descriptor1), (category2, descriptor2), (category3, descriptor3), (category4, descriptor4), (category5, descriptor5)).
                """
            },
            {
                "role": "user",
                "content": "\n".join([inner_list[1] for inner_list in image_urls])
            }
        ],
        temperature=0.2
    )
    output = response.choices[0].message.content
    return extract_tuples(output)


def get_tags(image_class, existing, descriptor, definition):
    """Get tags from OpenAI synchronously"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Give a list of 10 or less possible common (descriptor: {descriptor}) of {image_class}. Here is the definition of (descriptor: {descriptor}): {definition}.
                The list should not overlap with the existing lists: {existing}. Present in the following form: (descriptor, [list of 10]).
                """
            }
        ],
        temperature=0.8
    )
    return response.choices[0].message.content
