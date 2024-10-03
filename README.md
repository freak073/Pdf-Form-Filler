# Pdf-Form-Filler
used bert-base-uncased model from Hugging Face to make a PDF Form Filler
And Also uses The Rule based Approach
## Features

- Upload a PDF form and a knowledge base file.
- Extract form fields from the PDF.
- Use BERT And Rules Based algorithm to map form fields to knowledge base entries.
- Fill the form fields in the PDF.
- Download the filled PDF.

## Requirements

- Python 3.7+
- Flask
- PyMuPDF (fitz)
- OpenCV
- NumPy
- pytesseract
- Pillow
- transformers
- torch
- scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/freak073/pdf-form-filler.git
   cd pdf-form-filler
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set the path to the Tesseract executable in `api.py`:

   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed
   ```

## Usage

1. Run the Flask application:

   ```bash
   python api.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/`.

3. Upload a PDF file and a knowledge base file using the form.

4. Click the "Upload and Process" button.

5. The filled PDF will be downloaded automatically.

## Project Structure
pdf-form-filler/
├── project/
│ ├── api.py
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ ├── css/
│ │ │ └── styles.css
│ │ ├── js/
│ │ │ └── scripts.js
│ └── output_images/
├── requirements.txt
└── README.md

## Code Overview

### `api.py`

This is the main Flask application file. It contains the following key functions:

- `parse_knowledge_base(file_path)`: Parses the knowledge base file.
- `extract_form_fields_from_pdf(pdf_path)`: Extracts form fields from the PDF.
- `encode_text(text)`: Encodes text using BERT.
- `map_fields_to_knowledge_base(form_fields, knowledge_base)`: Maps form fields to knowledge base entries using BERT.
- `pdf_to_images(pdf_path, output_dir)`: Converts PDF pages to images.
- `preprocess_image(image)`: Preprocesses the image for form field detection.
- `detect_cells(gray)`: Detects cells in the preprocessed image.
- `is_cell_empty(img, x, y, w, h)`: Checks if a cell is empty.
- `get_field_name(img, x, y, w, h)`: Gets the field name from the image.
- `put_text_in_box(img, text, x, y, w, h, color, align_left, align_top)`: Puts text in a box in the image.
- `format_text_as_in_pdf(text, field_width, field_height, max_font_size, min_font_size)`: Formats text to fit in the PDF field.
- `detect_main_products_blocks(img)`: Detects main products blocks in the image.
- `fill_main_products(img, text, blocks)`: Fills main products blocks in the image.
- `image_to_pdf(image_path, output_pdf_path)`: Converts an image to a PDF.
- `process_pdf()`: The main endpoint for processing the PDF and returning the filled PDF.

### `index.html`

The main HTML file for the web interface. It contains a form for uploading the PDF and knowledge base files.

### `styles.css`

The CSS file for styling the web interface.

### `scripts.js`

The JavaScript file for handling the form submission and file download.


