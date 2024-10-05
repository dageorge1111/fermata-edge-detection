from flask import Flask, request, jsonify
from supabase import create_client, Client
import openai
import os
import re
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
supabase_key = os.getenv('SUPABASE_KEY')
supabase_url = os.getenv('SUPABASE_URL')

# Set up Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

openai.api_key = os.getenv('OPENAI_KEY')

def extract_tuples(input_string):
    """Extract tuples from the input string"""
    tuple_regex = r"\(([^)]+)\)"
    matches = re.findall(tuple_regex, input_string)
    return [tuple(map(str.strip, match.split(','))) for match in matches]

async def analyze_segmented_image(image_class, *image_urls):
    """Analyze segmented images using OpenAI"""
    response = await openai.ChatCompletion.acreate(
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
                "content": "\n".join(image_urls) 
            }
        ],
        temperature=0.2
    )
    output = response.choices[0].message['content']
    return extract_tuples(output)

async def get_tags(image_class, existing, descriptor, definition):
    """Get tags from OpenAI"""
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Give a list of 10 or less possible common {descriptor} of {image_class}. Here is the definition of {descriptor}: {definition}. 
                The list should not overlap with the existing lists: {existing}. Present as comma-separated.
                """
            }
        ],
        temperature=0.8
    )
    return response.choices[0].message['content']
