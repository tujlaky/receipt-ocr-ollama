import sys
import ollama
import argparse

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


def extract_text_from_image(image_path, model):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
        keep_alive="30m",
        options={
            "num_ctx": 2048,
            "num_predict": 350,  # plenty for compact JSON
            "temperature": 0,
            # try increasing threads if you like: "num_thread": 8,
        },
        format="json",
    )

    return response["message"]["content"]


parser = argparse.ArgumentParser(
    description="Extract information from a receipt with ollama"
)

parser.add_argument("image_path", type=str, help="The path to the receipt")
parser.add_argument(
    "--model", type=str, default=DEFAULT_MODEL, help="Default model to use"
)

args = parser.parse_args()

text = extract_text_from_image(args.image_path, args.model)
print(text)
