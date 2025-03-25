import requests
import pytesseract
import json
import os
import argparse
from PIL import Image
from gemma import gm

# Load Gemma model and parameters
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
sampler = gm.text.ChatSampler(model=model, params=params, multi_turn=True)

def extract_text_from_image(image_path):
    """ Extracts text from an image using Tesseract OCR """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def query_gemma_api(prompt):
    """ Queries the locally loaded Gemma model with the extracted text """
    response = sampler.chat(prompt)
    return response

def parse_markdown_table(markdown_table):
    """ Converts a Markdown table into a structured JSON object while ensuring data integrity """
    lines = [line.strip() for line in markdown_table.split("\n") if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and "|" in line]
    if len(table_lines) < 2:
        return None  # Not a valid table

    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    data_rows = [
        [cell.strip().strip('"') for cell in row.split("|") if cell.strip()]  # Remove extra quotes
        for row in table_lines[2:]  # Skip header separator line
    ]

    json_data = []
    for row in data_rows:
        row_data = {}
        for i in range(len(headers)):
            value = row[i] if i < len(row) else ""
            # Convert numeric values properly, handling missing values
            if value.replace('.', '', 1).isdigit():
                row_data[headers[i]] = float(value) if '.' in value else int(value)
            else:
                row_data[headers[i]] = value
        json_data.append(row_data)

    return json_data if json_data else None

def save_json(data, output_directory, filename):
    """ Saves extracted table data as a JSON file in the output directory """
    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists
    output_path = os.path.join(output_directory, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Extracted table data saved as {output_path}")

def process_images_in_directory(input_directory, output_directory):
    """ Process all images in a directory and save each extracted table data into a separate JSON file """
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_directory, filename)
            print(f"Processing: {image_path}")

            # Step 1: Extract text from image
            extracted_text = extract_text_from_image(image_path)

            # Step 2: Construct a prompt for the LLM
            prompt = f"Extract only the table data in Markdown format from the following text:\n{extracted_text}"

            # Step 3: Send to Gemma API
            generated_text = query_gemma_api(prompt)

            print("\nGenerated Table in Markdown:\n", generated_text)

            # Step 4: Convert Markdown table to JSON
            table_json = parse_markdown_table(generated_text)

            if table_json:
                output_filename = f"{os.path.splitext(filename)[0]}.json"
                save_json(table_json, output_directory, output_filename)
            else:
                print(f"\nError: Unable to parse Markdown table from {filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from images and save them as JSON.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing images")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory where JSON files will be saved")
    args = parser.parse_args()

    process_images_in_directory(args.input_dir, args.output_dir)
