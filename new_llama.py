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
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf > 70 and text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            tokens.append({
                'text': text,
                'bbox': [x, y, x + w, y + h]
            })
    return tokens

def tokens_to_text_prompt(tokens):
    layout = ""
    for token in tokens:
        x1, y1, x2, y2 = token['bbox']
        layout += f"[{x1},{y1},{x2},{y2}]: {token['text']}\n"
    return layout

def prompt_llama(prompt: str, retries=2) -> str:
    for _ in range(retries):
        response = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
        )
        result = response.stdout.decode('utf-8')
        if '|' in result:
            return result
    return result

def recognize_table_with_llama(image: Image.Image, tokens):
    token_prompt = tokens_to_text_prompt(tokens)
    prompt = f"""
You are a table recognition engine.

Your task is to reconstruct a table from OCR tokens.
Each token is in the format:
[x1,y1,x2,y2]: word

First, identify the **table header row** — typically the top row with column names. Then, use the position of each header to define columns.
Assign each following token to a column based on horizontal alignment with the headers.

Return the table in GitHub-flavored **Markdown format** only.
Do not include explanations or comments.

Tokens:
{token_prompt}
"""
    return prompt_llama(prompt)

def markdown_to_csv(md: str) -> str:
    lines = [line for line in md.splitlines() if '|' in line and not line.strip().startswith('|---')]
    rows = [line.strip().strip('|').split('|') for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    header = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []
    df = pd.DataFrame(data, columns=header)
    return df.to_csv(index=False)

def html_from_tokens(tokens):
    sorted_tokens = sorted(tokens, key=lambda t: (t['bbox'][1], t['bbox'][0]))
    html = "<div style='position:relative; width:100%;'>\n"
    for token in sorted_tokens:
        x1, y1, x2, y2 = token['bbox']
        width = x2 - x1
        height = y2 - y1
        style = (
            f"position:absolute; left:{x1}px; top:{y1}px; width:{width}px; height:{height}px; "
            f"padding:2px; font-size:12px; background:white;"
        )
        html += f"  <div style=\"{style}\">{token['text']}</div>\n"
    html += "</div>"
    return html

def save_html_with_style(html: str, path: str):
    styled_html = f"""
<html>
<head>
<style>
body {{ background-color: #f9f9f9; }}
</style>
</head>
<body>
{html}
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(styled_html)

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
        print(f"\n[\u2713] Processed: {image_file}\n")

        csv = markdown_to_csv(markdown)
        with open(output_base + "_table.csv", "w") as f:
            f.write(csv)

        html_bbox = html_from_tokens(tokens)
        save_html_with_style(html_bbox, output_base + "_table.html")

        visualize_cells(image, tokens, output_base + "_tokens.jpg")

if __name__ == "__main__":
    main()
