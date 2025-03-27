# Before running script:
# Installing huggingface-cli: pip install huggingface_hub
# Logging in huggingface-cli: huggingface-cli login

import logging
import time
from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter
import argparse

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to directory containing images")
    parser.add_argument("--output_dir", required=True, help="Path to directory to store results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) # Change with your image directory path
    output_dir = Path(args.output_dir) # Change with your output path

    doc_converter = DocumentConverter()

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    for input_doc_path in sorted(input_dir.iterdir()):
        if not input_doc_path.suffix.lower() in image_extensions:
            continue  # Skip non-image files

        _log.info(f"Processing {input_doc_path.name}")
        start_time = time.time()

        try:
            conv_res = doc_converter.convert(input_doc_path)
        except Exception as e:
            _log.error(f"Failed to convert {input_doc_path.name}: {e}")
            continue

        doc_filename = input_doc_path.stem

        # Export tables
        for table_ix, table in enumerate(conv_res.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()
            print(f"## Table {table_ix}")
            print(table_df.to_markdown())

            # Save the table as csv
            element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix+1}.csv"
            _log.info(f"Saving CSV table to {element_csv_filename}")
            table_df.to_csv(element_csv_filename)

            # Save the table as html
            element_html_filename = output_dir / f"{doc_filename}-table-{table_ix+1}.html"
            _log.info(f"Saving HTML table to {element_html_filename}")
            with element_html_filename.open("w") as fp:
                fp.write(table.export_to_html())

    end_time = time.time() - start_time

    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
