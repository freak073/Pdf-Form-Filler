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
from difflib import SequenceMatcher

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Corrected model name
model = BertModel.from_pretrained('bert-base-uncased')  # Corrected model name

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

        # Print the field name and its mapped value
        print(f"Field: {field_name} -> Mapped Value: {best_match_value}")
    
    return field_mappings

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
    field_name = left_text.strip() if left_text.strip() else "Unknown Field"
    print(f"Extracted field name: {field_name}")  # Debugging output
    return field_name

def put_text_in_box(img, text, x, y, w, h, color=(0, 0, 0), font_size=38, thickness=2, align_left=False, align_top=False):
    # Convert OpenCV image to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    margin = 5
    line_spacing = 8  # Increased line spacing
    
    # Split text into lines if it's too wide
    lines = []
    words = text.split()
    if words:  # Check if words list is not empty
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

def find_best_match(field_name, field_mappings):
    best_match = None
    highest_similarity = 0.0
    
    for key in field_mappings.keys():
        similarity = SequenceMatcher(None, field_name.lower(), key.lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = key
    
    return best_match if highest_similarity > 0.6 else None  # Adjust threshold as needed

def fill_text_using_field_name(img, field_name, x, y, w, h, field_mappings, font_size=28):
    """
    Fill the text in the specified area using the given field name's mapped value.
    """
    matched_key = find_best_match(field_name, field_mappings)
    
    if matched_key:
        mapped_value = field_mappings[matched_key]
        print(f"Filling field '{field_name}' with matched key '{matched_key}' and value: {mapped_value}")  # Debugging output
    else:
        mapped_value = f"{{{field_name}}}"
        print(f"Field '{field_name}' not found in mappings. Using placeholder.")  # Debugging output
    
    img = put_text_in_box(img, mapped_value, x, y, w, h, align_left=True, align_top=True, font_size=font_size)
    return img

def detect_and_mark_cells(image_path, output_image_path, field_mappings):
    img = cv2.imread(image_path)
    
    # Preprocess the image
    preprocessed = preprocess_image(img)
    
    cells = detect_cells(preprocessed)
    
    # Process cells
    i = 0
    while i < len(cells):
        x, y, w, h = cells[i]
        if is_cell_empty(img, x, y, w, h):
            # Check for multiple empty cells
            multi_cells = [cells[i]]
            j = i + 1
            while j < len(cells) and cells[j][1] == y and is_cell_empty(img, *cells[j]):
                multi_cells.append(cells[j])
                j += 1
            
            if len(multi_cells) > 1:
                # Handle multi-value inputs
                field_name = get_field_name(img, multi_cells[0][0], multi_cells[0][1], multi_cells[0][2], multi_cells[0][3])
                
                # Reverse the order of multi_cells to process them from left to right
                multi_cells.reverse()
                
                # Split the mapped value into sentences
                mapped_value = field_mappings.get(field_name, f"{{{field_name}}}")
                sentences = mapped_value.split(',') if mapped_value else []
                
                for idx, cell in enumerate(multi_cells, 1):
                    x, y, w, h = cell
                    # Draw a rectangle around the cell
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Green rectangle
                    fill_text = sentences[idx - 1] if idx - 1 < len(sentences) else ""
                    img = put_text_in_box(img, fill_text, x, y, w, h, font_size=24, align_left=True)  # Adjusted font size for multi-cells
                i = j - 1
            else:
                # Draw a rectangle around the cell
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Green rectangle
                field_name = get_field_name(img, x, y, w, h)
                img = fill_text_using_field_name(img, field_name, x, y, w, h, field_mappings, font_size=28)  # Use the new function
        else:
            # Draw a rectangle around non-empty cells
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Red rectangle
        i += 1

    # Process entire image for Q. and A.
    pil_img = Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    
    i = 0
    last_question = ""
    while i < len(data['text']):
        if data['text'][i].strip().lower().startswith('q.'):
            start_x, start_y = data['left'][i], data['top'][i]
            end_x, end_y = start_x + data['width'][i], start_y + data['height'][i]
            
            # Find the end of the question (next Q. or A.)
            j = i + 1
            while j < len(data['text']) and not (data['text'][j].strip().lower().startswith('q.') or data['text'][j].strip().lower().startswith('a.')):
                end_x = max(end_x, data['left'][j] + data['width'][j])
                end_y = max(end_y, data['top'][j] + data['height'][j])
                j += 1
            
            # Adjust the question box size
            start_x = max(0, start_x - 5)
            start_y = max(0, start_y - 5)
            end_x = min(img.shape[1], end_x + 5)
            end_y = min(img.shape[0], end_y - int((end_y - start_y) * 0.2))  # Reduce height by 20%
            
            # Draw a rectangle around the question
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)  # Blue rectangle
            last_question = ' '.join(data['text'][i:j])
            i = j - 1  # Move to the last processed word
        
        elif data['text'][i].strip().lower().startswith('a.'):
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            
            # Create imaginary box that includes A. and extends below
            img_h, img_w = img.shape[:2]
            answer_box_w = int((img_w - (x + w + 3)) * 0.9)  # 10% shorter from the right side
            
            # Find the next element's y-coordinate
            next_element_y = img_h
            for j in range(i+1, len(data['text'])):
                if data['text'][j].strip():
                    next_element_y = data['top'][j]
                    break
            
            answer_box_h = next_element_y - y - 10  # Leave a small gap
            answer_box_y = y
            # Fill answer box with mapped value
            field_name = last_question.split(':')[0].strip()  # Extract field name from the last question
            img = fill_text_using_field_name(img, field_name, x + w + 3, answer_box_y, answer_box_w, answer_box_h, field_mappings, font_size=28)  # Use the new function
        
        i += 1

    cv2.imwrite(output_image_path, img)
    return img


def process_pdf(input_pdf, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    print("Converting PDF to images...")
    image_paths = pdf_to_images(input_pdf, output_dir)

    # Extract form fields from the PDF
    form_fields = extract_form_fields_from_pdf(input_pdf)

    # Parse the knowledge base
    knowledge_base_path = "dummy_data.txt"
    knowledge_base = parse_knowledge_base(knowledge_base_path)

    # Map form fields to knowledge base entries using BERT
    field_mappings = map_fields_to_knowledge_base(form_fields, knowledge_base)


    # Process each page
    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i + 1}...")
        output_image_path = os.path.join(output_dir, f'marked_page_{i + 1}.png')
        detect_and_mark_cells(image_path, output_image_path, field_mappings)
        print(f"Marked image for page {i + 1} saved to: {output_image_path}")

        # Clean up temporary image file
        os.remove(image_path)

    print("PDF processing complete.")

# Get the current script's directory
script_dir = os.getcwd()

# Construct the paths
input_pdf = os.path.join(script_dir, 'Dummy_Questionnaire.pdf')
output_dir = os.path.join(script_dir, 'output')

# Process the PDF
process_pdf(input_pdf, output_dir)

def images_to_pdf(image_paths, output_pdf_path):
    # Create a new PDF document
    pdf_document = fitz.open()

    for image_path in image_paths:
        # Open the image using PIL
        img = Image.open(image_path)
        
        # Convert the image to RGB if it's not
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Save the image to a temporary file in PDF format
        temp_pdf_path = image_path.replace(".png", ".pdf")
        img.save(temp_pdf_path, "PDF", resolution=100.0)
        
        # Open the temporary PDF file and add it to the document
        temp_pdf = fitz.open(temp_pdf_path)
        pdf_document.insert_pdf(temp_pdf)
        
        # Close the temporary
        

        