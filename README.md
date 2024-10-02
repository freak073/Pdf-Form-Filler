# Pdf-Form-Filler
used bert-base-uncased model from Hugging Face to make a PDF Form Filler
## Features

- Upload a PDF form and a knowledge base file.
- Extract form fields from the PDF.
- Use BERT to map form fields to knowledge base entries.
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
