import os
import base64
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get image description
def get_image_description(image_path, title):
  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": 'YOUR_GPT4o_API_KEY'
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": title
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
              "detail": "low"
            }
          }
        ]
      }
    ],
    "max_tokens": 1800
  }



  response = requests.post("https://api.kwwai.top/v1/chat/completions", headers=headers, json=payload)

  if response.status_code != 200:
    print(response.text)
    return None

  # Extract the description from the response
  if 'content' in response.json()['choices'][0]['message']:
    description = response.json()['choices'][0]['message']['content']
  else:
    description = None
  return description
