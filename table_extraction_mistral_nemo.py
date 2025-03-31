import requests
import pytesseract
import json
from PIL import Image

def extract_text_from_image(image_path):
    """ Extracts text from an image using Tesseract OCR """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def query_huggingface_api(prompt, model_name="mistralai/Mistral-Nemo-Instruct-2407", hf_token="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"): # Change with your hugging face token
    """ Queries Hugging Face API with the extracted text """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.01,  # Reduce randomness for structured output
        },
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def parse_markdown_table(markdown_table):
    """ Converts a Markdown table into a JSON object """
    lines = [line.strip() for line in markdown_table.split("\n") if line.strip()]
    if len(lines) < 3:
        return []  # Not a valid table
    
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    data_rows = [
        [cell.strip() for cell in row.split("|") if cell.strip()] 
        for row in lines[2:]  # Skip header separator
    ]
    
    json_data = [dict(zip(headers, row)) for row in data_rows if len(row) == len(headers)]
    return json_data

def save_json(data, filename="output.json"):
    """ Saves extracted table data as a JSON file """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Provide the image path
image_path = "/home/unknown/Documents/table_image/input/cropped/table1.png"  # Change this to the path of your image

# Step 1: Extract text from image
extracted_text = extract_text_from_image(image_path)

# Step 2: Construct a prompt for the LLM
prompt = f"Here is text that contains a table or multiple tables:\n{extracted_text}\n\nPlease extract the table in Markdown format."

# Step 3: Send to Hugging Face API
result = query_huggingface_api(prompt)

# Step 4: Extract generated text from response
if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
    generated_text = result[0]["generated_text"]
    print("\nGenerated Table in Markdown:\n", generated_text)
    
    # Step 5: Convert Markdown table to JSON
    table_json = parse_markdown_table(generated_text)
    
    if table_json:
        save_json(table_json)
        print("\nExtracted table data saved as output.json")
    else:
        print("\nError: Unable to parse Markdown table.")
else:
    print("\nError: Unable to extract generated text.")
