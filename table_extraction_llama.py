import subprocess
import json
from PIL import Image
import base64
import io
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from collections import OrderedDict
import pandas as pd
import os
import argparse

def run_ocr(image: Image.Image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            tokens.append({
                'text': data['text'][i],
                'bbox': [x, y, x + w, y + h]
            })
    return tokens

def tokens_to_text_prompt(tokens):
    layout = ""
    for token in tokens:
        x1, y1, x2, y2 = token['bbox']
        layout += f"[{x1},{y1},{x2},{y2}]: {token['text']}\n"
    return layout

def prompt_llama(prompt: str) -> str:
    response = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode('utf-8'),
        stdout=subprocess.PIPE,
    )
    return response.stdout.decode('utf-8')

def recognize_table_with_llama(image: Image.Image, tokens):
    token_prompt = tokens_to_text_prompt(tokens)
    prompt = f"""
You are a table recognition engine. Below is the OCR token data extracted from a table image.
Each line is a token with its bounding box in the format:
[x1,y1,x2,y2]: word
Based on these tokens, reconstruct the table in Markdown format.

Tokens:
{token_prompt}

Return only the Markdown table.
"""
    markdown_table = prompt_llama(prompt)
    return markdown_table

def markdown_to_csv(md: str) -> str:
    lines = [line for line in md.splitlines() if '|' in line and not line.strip().startswith('|---')]
    rows = [line.strip().strip('|').split('|') for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    header = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []
    df = pd.DataFrame(data, columns=header)
    return df.to_csv(index=False)

def markdown_to_html(md: str) -> str:
    lines = [line for line in md.splitlines() if '|' in line and not line.strip().startswith('|---')]
    rows = [line.strip().strip('|').split('|') for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    html = "<table>\n"
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        html += "  <tr>" + ''.join([f"<{tag}>{cell}</{tag}>" for cell in row]) + "</tr>\n"
    html += "</table>"
    return html

def visualize_cells(image: Image.Image, tokens, out_path):
    plt.imshow(image, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()
    for token in tokens:
        bbox = token['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        input_path = os.path.join(args.input_dir, image_file)
        output_base = os.path.splitext(os.path.join(args.output_dir, image_file))[0]

        image = Image.open(input_path).convert("RGB")
        tokens = run_ocr(image)
        markdown = recognize_table_with_llama(image, tokens)
        print(f"\n[✓] Processed: {image_file}\n")

        csv = markdown_to_csv(markdown)
        html = markdown_to_html(markdown)

        with open(output_base + "_table.csv", "w") as f:
            f.write(csv)
        with open(output_base + "_table.html", "w") as f:
            f.write(html)
        visualize_cells(image, tokens, output_base + "_tokens.jpg")

if __name__ == "__main__":
    main()