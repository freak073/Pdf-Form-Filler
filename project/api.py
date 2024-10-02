from flask import Flask, request, jsonify, render_template, send_file
import fitz  # PyMuPDF for PDF processing
import re  # For knowledge base parsing
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')

# Step 1: Parse the Knowledge Base (dummy_data.txt)
def parse_knowledge_base(file_path):
    knowledge_base = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.match(r'(.+?):\s*(.+)', line)
            if match:
                key, value = match.groups()
                knowledge_base[key.strip()] = value.strip()
    return knowledge_base

# Step 2: Extract Form Fields from PDF
def extract_form_fields_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    form_fields = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) >= 5:
                x0, y0, x1, y1 = block[:4]
                field_text = block[4].strip()
                if field_text:
                    form_fields.append({
                        "page": page_num + 1,
                        "coordinates": (x0, y0, x1, y1),
                        "field_name": field_text
                    })
    return form_fields

# Step 3: Use BERT to encode text
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling on token embeddings
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings

# Step 4: Map Form Fields to Knowledge Base using BERT
def map_fields_to_knowledge_base(form_fields, knowledge_base):
    field_mappings = {}
    knowledge_base_keys = list(knowledge_base.keys())
    
    # Encode all knowledge base keys
    kb_embeddings = torch.cat([encode_text(key) for key in knowledge_base_keys])

    for form_field in form_fields:
        field_name = form_field['field_name']
        form_embedding = encode_text(field_name)
        
        # Reshape embeddings to 2D arrays
        form_embedding_2d = form_embedding.numpy().reshape(1, -1)
        kb_embeddings_2d = kb_embeddings.numpy()
        
        # Calculate cosine similarity between form field and knowledge base keys
        similarities = cosine_similarity(form_embedding_2d, kb_embeddings_2d)
        
        # Find the best match based on the highest similarity score
        best_match_idx = similarities.argmax()
        best_match_key = knowledge_base_keys[best_match_idx]
        best_match_value = knowledge_base[best_match_key]
        
        # Store the best match
        field_mappings[field_name] = best_match_value

    return field_mappings

# Step 5: Process the PDF for filling form fields
def pdf_to_images(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Increase resolution
        img_path = os.path.join(output_dir, f'temp_page_{page_num + 1}.png')
        pix.save(img_path)
        images.append(img_path)
    doc.close()
    return images

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return denoised

def detect_cells(gray):
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3000 < area < 200000:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 20:
                cells.append((x, y, w, h))
    return cells

def is_cell_empty(img, x, y, w, h):
    cell = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(binary == 255) / (w * h)
    return white_pixel_ratio > 0.95

def get_field_name(img, x, y, w, h):
    # Check the entire left side for the field name
    left_cell = img[y:y+h, 0:x]
    left_text = pytesseract.image_to_string(left_cell)
    return left_text.strip() if left_text.strip() else "Unknown Field"

def put_text_in_box(img, text, x, y, w, h, color=(0, 0, 0), align_left=False, align_top=False):
    # Convert OpenCV image to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Format the text to fit the field
    formatted_text, font_size = format_text_as_in_pdf(text, w, h)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    margin = 5
    line_spacing = 8  # Increased line spacing
    
    # Split text into lines if it's too wide
    lines = []
    words = formatted_text.split()
    current_line = words[0]
    for word in words[1:]:
        bbox = draw.textbbox((0, 0), current_line + " " + word, font=font)
        if bbox[2] - bbox[0] <= w - 2*margin:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    # Calculate total text height
    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = bbox[3] - bbox[1] + line_spacing
    total_text_height = len(lines) * line_height - line_spacing
    
    # Draw text
    for i, line in enumerate(lines):
        if align_top:
            text_y = y + margin + i * line_height
        else:
            text_y = y + (h - total_text_height) // 2 + i * line_height
        
        bbox = draw.textbbox((0, 0), line, font=font)
        if align_left:
            text_x = x + margin
        else:
            text_x = x + (w - (bbox[2] - bbox[0])) // 2
        
        # Draw text with a slight offset to create a bold effect
        for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), line, font=font, fill=color)
    
    # Convert back to OpenCV image
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def format_text_as_in_pdf(text, field_width, field_height, max_font_size=38, min_font_size=20):
    from PIL import ImageFont, ImageDraw, Image
    
    # Create a dummy image to calculate text size
    dummy_img = Image.new('RGB', (field_width, field_height))
    draw = ImageDraw.Draw(dummy_img)
    
    # Start with the maximum font size and decrease until the text fits
    font_size = max_font_size
    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Calculate the size of the text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Check if the text fits within the field
        if text_width <= field_width and text_height <= field_height:
            break
        
        # Decrease the font size
        font_size -= 1
    
    return text, font_size

def detect_main_products_blocks(img):
    # Assuming the main products block is divided into three sub-blocks
    # You need to detect these sub-blocks manually or through some heuristic
    # For simplicity, let's assume we have the coordinates of these sub-blocks
    # Replace these coordinates with the actual ones from your PDF
    main_products_blocks = [
        (100, 200, 150, 50),  # Coordinates of the first sub-block (x, y, width, height)
        (300, 200, 150, 50),  # Coordinates of the second sub-block
        (500, 200, 150, 50)   # Coordinates of the third sub-block
    ]
    return main_products_blocks

def fill_main_products(img, text, blocks):
    words = text.split()
    for i, block in enumerate(blocks):
        if i < len(words):
            x, y, w, h = block
            img = put_text_in_box(img, words[i], x, y, w, h)
    return img

# New function to convert a single image to PDF
def image_to_pdf(image_path, output_pdf_path):
    image = Image.open(image_path)
    pdf_bytes = image.convert('RGB')
    pdf_bytes.save(output_pdf_path, "PDF", resolution=100.0)
    print(f"PDF saved: {output_pdf_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    pdf = request.files['pdf']
    knowledge_base = request.files['knowledge_base']
    output_dir = "output_images"
    
    # Save uploaded files
    pdf_path = os.path.join(output_dir, pdf.filename)
    knowledge_base_path = os.path.join(output_dir, knowledge_base.filename)
    pdf.save(pdf_path)
    knowledge_base.save(knowledge_base_path)
    
    # Extract form fields from the PDF
    form_fields = extract_form_fields_from_pdf(pdf_path)

    # Parse the knowledge base
    knowledge_base = parse_knowledge_base(knowledge_base_path)

    # Map form fields to knowledge base entries using BERT
    field_mappings = map_fields_to_knowledge_base(form_fields, knowledge_base)
    
    # Convert PDF to images
    os.makedirs(output_dir, exist_ok=True)
    image_paths = pdf_to_images(pdf_path, output_dir)
    
    for page_num, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        preprocessed = preprocess_image(img)
        cells = detect_cells(preprocessed)
        
        # Fill the form fields in the image sequentially
        for field_name, fill_text in field_mappings.items():
            if field_name == "main products":
                main_products_blocks = detect_main_products_blocks(img)
                img = fill_main_products(img, fill_text, main_products_blocks)
            else:
                for cell in cells:
                    x, y, w, h = cell
                    if is_cell_empty(img, x, y, w, h):
                        current_field_name = get_field_name(img, x, y, w, h)
                        if current_field_name == field_name:
                            # Ensure the text is formatted as shown in Dummy_Questionnaire_Desired_Output.pdf
                            formatted_text, font_size = format_text_as_in_pdf(fill_text, w, h)  # Pass w and h
                            img = put_text_in_box(img, formatted_text, x, y, w, h)
                            if field_name == "Q. What is WECOMMIT dummy Inc.'s specific target year for achieving carbon neutrality, and what does this goal entail?":
                                # Insert the mapped value below the question
                                img = put_text_in_box(img, fill_text, x, y + h + 10, w, h)  # Adjust y-coordinate to place below
                            # Draw the field name and its mapped value
                            img = put_text_in_box(img, f"{field_name}: {fill_text}", x, y + h + 20, w, h)  # Adjust y-coordinate to place below
                            break  # Move to the next field in the sequence
        
        # Save the filled image
        output_image_path = os.path.join(output_dir, f"filled_page_{page_num + 1}.png")
        cv2.imwrite(output_image_path, img)
        print(f"Filled page saved: {output_image_path}")
        
        # Convert the filled image to PDF
        output_pdf_path = os.path.join(output_dir, f"filled_page_{page_num + 1}.pdf")
        image_to_pdf(output_image_path, output_pdf_path)

    # Return the first generated PDF file for download
    return send_file(output_pdf_path, as_attachment=True, mimetype='application/pdf')

if __name__ == "__main__":
    app.run(debug=True)