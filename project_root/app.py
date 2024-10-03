from flask import Flask, request, send_file, render_template
import os
import fitz  # PyMuPDF for PDF processing
import re  # For knowledge base parsing
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed

app = Flask(__name__, static_folder='static')

# PDF processing functions
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

def rule_based_mapping(form_fields, knowledge_base):
    field_mappings = {}
    for field in form_fields:
        form_field_name = field['field_name'].lower()
        for key, value in knowledge_base.items():
            key_lower = key.lower()
            if key_lower in form_field_name or form_field_name in key_lower:
                field_mappings[field['field_name']] = value
                break
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
    left_cell = img[y:y+h, 0:x]
    left_text = pytesseract.image_to_string(left_cell)
    field_name = left_text.strip() if left_text.strip() else "Unknown Field"
    return field_name

def put_text_in_box(img, text, x, y, w, h, color=(0, 0, 0), font_size=38, thickness=2, align_left=False, align_top=False):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    margin = 5
    line_spacing = 8
    
    lines = []
    words = text.split()
    if words:
        current_line = words[0]
        for word in words[1:]:
            bbox = draw.textbbox((0, 0), current_line + " " + word, font=font)
            if bbox[2] - bbox[0] <= w - 2*margin:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    
    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = bbox[3] - bbox[1] + line_spacing
    total_text_height = len(lines) * line_height - line_spacing
    
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
        
        for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), line, font=font, fill=color)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def find_best_match(field_name, field_mappings):
    best_match = None
    highest_similarity = 0.0
    
    for key in field_mappings.keys():
        similarity = SequenceMatcher(None, field_name.lower(), key.lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = key
    
    return best_match if highest_similarity > 0.6 else None

def fill_text_using_field_name(img, field_name, x, y, w, h, field_mappings, font_size=28):
    matched_key = find_best_match(field_name, field_mappings)
    
    if matched_key:
        mapped_value = field_mappings[matched_key]
    else:
        mapped_value = f"{{{field_name}}}"
    
    img = put_text_in_box(img, mapped_value, x, y, w, h, align_left=True, align_top=True, font_size=font_size)
    return img

def detect_and_mark_cells(image_path, output_image_path, field_mappings):
    img = cv2.imread(image_path)
    preprocessed = preprocess_image(img)
    cells = detect_cells(preprocessed)
    
    i = 0
    while i < len(cells):
        x, y, w, h = cells[i]
        if is_cell_empty(img, x, y, w, h):
            multi_cells = [cells[i]]
            j = i + 1
            while j < len(cells) and cells[j][1] == y and is_cell_empty(img, *cells[j]):
                multi_cells.append(cells[j])
                j += 1
            
            if len(multi_cells) > 1:
                field_name = get_field_name(img, multi_cells[0][0], multi_cells[0][1], multi_cells[0][2], multi_cells[0][3])
                multi_cells.reverse()
                mapped_value = field_mappings.get(field_name, f"{{{field_name}}}")
                sentences = mapped_value.split(',') if mapped_value else []
                
                for idx, cell in enumerate(multi_cells, 1):
                    x, y, w, h = cell
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    fill_text = sentences[idx - 1] if idx - 1 < len(sentences) else ""
                    img = put_text_in_box(img, fill_text, x, y, w, h, font_size=24, align_left=True)
                i = j - 1
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                field_name = get_field_name(img, x, y, w, h)
                img = fill_text_using_field_name(img, field_name, x, y, w, h, field_mappings, font_size=28)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        i += 1

    pil_img = Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    
    i = 0
    last_question = ""
    while i < len(data['text']): 
        if data['text'][i].strip().lower().startswith('q.'):
            start_x, start_y = data['left'][i], data['top'][i]
            end_x, end_y = start_x + data['width'][i], start_y + data['height'][i]
            
            j = i + 1
            while j < len(data['text']) and not (data['text'][j].strip().lower().startswith('q.') or data['text'][j].strip().lower().startswith('a.')):
                end_x = max(end_x, data['left'][j] + data['width'][j])
                end_y = max(end_y, data['top'][j] + data['height'][j])
                j += 1
            
            start_x = max(0, start_x - 5)
            start_y = max(0, start_y - 5)
            end_x = min(img.shape[1], end_x + 5)
            end_y = min(img.shape[0], end_y - int((end_y - start_y) * 0.2))
            
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
            last_question = ' '.join(data['text'][i:j])
            i = j - 1
        
        elif data['text'][i].strip().lower().startswith('a.'):
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            
            img_h, img_w = img.shape[:2]
            answer_box_w = int((img_w - (x + w + 3)) * 0.9)
            
            next_element_y = img_h
            for j in range(i+1, len(data['text'])):
                if data['text'][j].strip():
                    next_element_y = data['top'][j]
                    break
            
            answer_box_h = next_element_y - y - 10
            answer_box_y = y
            field_name = last_question.split(':')[0].strip()
            img = fill_text_using_field_name(img, field_name, x + w + 3, answer_box_y, answer_box_w, answer_box_h, field_mappings, font_size=28)
        
        i += 1

    cv2.imwrite(output_image_path, img)
    return img

def images_to_pdf(image_paths, output_pdf_path):
    pdf_document = fitz.open()

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            temp_pdf_path = image_path.replace(".png", ".pdf")
            img.save(temp_pdf_path, "PDF", resolution=100.0)
            
            temp_pdf = fitz.open(temp_pdf_path)
            pdf_document.insert_pdf(temp_pdf)
            temp_pdf.close()
            os.remove(temp_pdf_path)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue

    try:
        pdf_document.save(output_pdf_path)
        pdf_document.close()
        logging.info(f"PDF successfully saved to {output_pdf_path}")
    except Exception as e:
        logging.error(f"Error saving PDF: {e}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Ensure the uploads directory exists
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    
    pdf_file = request.files['pdf']
    knowledge_base_file = request.files['knowledge_base']
    
    pdf_path = os.path.join(uploads_dir, pdf_file.filename)
    knowledge_base_path = os.path.join(uploads_dir, knowledge_base_file.filename)
    pdf_file.save(pdf_path)
    knowledge_base_file.save(knowledge_base_path)
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = pdf_to_images(pdf_path, output_dir)
    form_fields = extract_form_fields_from_pdf(pdf_path)
    knowledge_base = parse_knowledge_base(knowledge_base_path)
    field_mappings = rule_based_mapping(form_fields, knowledge_base)
    
    for i, image_path in enumerate(image_paths):
        output_image_path = os.path.join(output_dir, f'mark_page_{i + 1}.png')
        detect_and_mark_cells(image_path, output_image_path, field_mappings)
        os.remove(image_path)
    
    output_pdf_path = r'c:\Users\varun\Desktop\Rule Based & Bert\project_root\output\final_output.pdf'
    images_to_pdf([os.path.join(output_dir, f'mark_page_{i + 1}.png') for i in range(len(image_paths))], output_pdf_path)
    
    # Check if the file exists before sending
    if not os.path.exists(output_pdf_path):
        logging.error(f"File not found: {output_pdf_path}")
        return "Error: File not found", 404
    
    return send_file(output_pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)