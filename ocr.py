import sys
import ollama
import argparse
from PIL import Image
import tempfile
import os


DEFAULT_MODEL = "llama3.2-vision"

prompt = """
Extract all text from this receipt image and return it strictly as a JSON object without any explanation
with the following structure:

{
  "merchant": {
    "name": "string",
    "address": "string",
    "phone": "string"
  },
  "items": [
    {
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "total": number
    }
  ],
  "date": "ISO 8601 string",
  "total": {
    "vat": {
      "rate": "string",
      "amount": number
    },
    "amount": number,
    "currency": "string"
  }
}

Rules:
- You should never change the JSON fields or formats
- Always respond with valid JSON only, no explanations.
- Don't return anything else just the JSON without wrappers
- If some fields are missing in the receipt, set their value to null.
"""


def preprocess_image(image_path, max_side=1280):
    img = Image.open(image_path)
    width, height = img.size
    scale = max(width, height) / max_side

    if scale > 1.0:
        img = img.resize((int(width / scale), int(height / scale)), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG", optimize=True)
        print(f"Image saved to: {tmp.name}")
        return tmp


def extract_text_from_image(image_path, model):
    tmp = preprocess_image(image_path, 1280)

    print(tmp.name)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [tmp.name],
            }
        ],
        keep_alive="30m",
        options={"num_ctx": 2048, "num_predict": 350, "temperature": 0},
        format="json",
    )

    os.unlink(tmp.name)

    return response["message"]["content"]


parser = argparse.ArgumentParser(
    description="Extract information from a receipt with ollama"
)

parser.add_argument("image_path", type=str, help="The path to the receipt")
parser.add_argument(
    "--model", type=str, default=DEFAULT_MODEL, help="Default model to use"
)

args = parser.parse_args()

print(f"Model: {args.model}")

text = extract_text_from_image(args.image_path, args.model)
print(text)
