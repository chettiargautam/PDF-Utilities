"""
Utility functions for PDF processing.

- `convert_pdf_to_images`: Converts a PDF file to a list of images.
- `pdf_to_html_pdf2txt`: Converts a PDF file to an HTML file using pdf2txt.
- `get_image_size`: Gets the dimensions of an image.
- `add_styles_to_hocr`: Adds CSS styles to the hocr content.
- `pdf_to_hocr_html_tesseract`: OCR on images using tesseract.
- `pdf_to_hocr_images_tesseract`: Converts PDF pages to images, performs OCR using Tesseract to obtain bounding boxes, and plots these boxes on the images.
- `convert_docx_to_pdf`: Convert a DOCX file to a PDF file.
- `convert_doc_to_pdf`: Convert a DOC file to a PDF file.
- `convert_pptx_to_pdf`: Convert a PPTX file to a PDF file.
- `split_pdf_pages`: Splits a PDF file into individual pages.
- `extract_ocr_data_from_pdf`: Extracts words, their bounding boxes, line IDs, paragraph IDs from a PDF using Tesseract OCR.
- `tag_pdf`: Makes a PDF file accessible.
- `extract_and_save_pdf_tags`: Extracts the tags from the PDF file and saves them as an HTML file.
- `parse_text`: Handle some of the unicode and other escape sequence issues.
- `image_to_pdf`: Converts an image file to a PDF file using Pillow.
- `extract_ocr_data_with_tags_from_pdf`: Extracts OCR data from PDF and augments it with HTML tag and style information.
"""
import os, json, typing


TEMP_IMAGE_DIR = "temp/images/"
HTML_TEMP_DIR = "temp/html/temp.html"
TAGS_TEMP_DIR = "temp/html"


def convert_pdf_to_images(pdf_file_path: str, output_save_dir: str = '') -> list:
    """
    Converts a PDF file to a list of images. The images are stored in the `output_save_dir`.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `output_save_dir` (str): Path to the directory where the images will be stored. Default is ''.

    Docs:
    - https://pypi.org/project/pdf2image/
    """
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_file_path)
    
    image_files = []
    if len(images) > 1:
        for i, image in enumerate(images):
            image_file_path = f"{output_save_dir}/{os.path.splitext(os.path.basename(pdf_file_path))[0]}_page_{i+1}.jpeg"
            if output_save_dir:
                image.save(image_file_path, "JPEG")
            image_files.append(image_file_path)
    if len(images) == 1:
        image_file_path = f"{output_save_dir}/{os.path.splitext(os.path.basename(pdf_file_path))[0]}.jpeg"
        if output_save_dir:
            images[0].save(image_file_path, "JPEG")
        image_files.append(image_file_path)
    return image_files


def pdf_to_html_pdf2txt(pdf_file_path: str, output_file_path: str, redact_anchor_tags: bool = True, visualize_bboxes: bool = True) -> None:
    """
    PDF to HTML conversion using pdf2txt. The output HTML file is saved at `output_file_path`. The anchor tags can be redacted and the bounding boxes can be visualized using flags.

    Args:
    - `pdf_file_path` (str): Path to the pdf file.
    - `output_file_path` (str): Path to the output html file.
    - `redact_anchor_tags` (bool): Flag to redact the anchor tags.
    - `visualize_bboxes` (bool): Flag to visualize the bounding boxes.

    Docs:
    - https://pdfminersix.readthedocs.io/en/latest/tutorial/commandline.html
    """
    os.system(f"pdf2txt.py -o {output_file_path} {pdf_file_path}")
    print(f">> HTML file {output_file_path} generated from {pdf_file_path} successfully.")

    if redact_anchor_tags:
        with open(output_file_path, "r") as f:
            lines = f.readlines()

        with open(output_file_path, "w") as f:
            for line in lines:
                if "</a>" not in line:
                    f.write(line)

    if visualize_bboxes:
        with open(output_file_path, "r") as f:
            html = f.read()
            html = html.replace("<head>", "<head><style>span {border: 1px solid red;} div {border: 1px solid green;}</style>")
        with open(output_file_path, "w") as f:
            f.write(html)


def get_image_size(image_path : str) -> tuple:
    """
    Gets the dimensions of an image.
    
    Args:
    - `image_path` (str): Path to the image file.
    
    Returns:
    - `dimensions` (tuple): A tuple containing the width and height of the image.
    """
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size


def add_styles_to_hocr(hocr_content : str) -> str:
    """
    Adds CSS styles to the hocr content. Useful for visualizing the bounding boxes.

    Args:
    - `hocr_content` (str): The hocr content.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(hocr_content, 'html.parser')
    lines = soup.find_all('span', class_='ocr_line')

    for line in lines:
        bbox_info = line['title'].split(';')
        coords = bbox_info[0].split()[1:]
        bbox = list(map(int, coords))
        line['style'] = f"border: 1px solid red; position: absolute; left: {bbox[0]}px; top: {bbox[1]}px; width: {bbox[2] - bbox[0]}px; height: {bbox[3] - bbox[1]}px;"

    return str(soup)


def pdf_to_hocr_html_tesseract(pdf_file_path: list, images_dir_path: str, output_hocr_file_dir: str) -> None:
    """
    OCR on images using tesseract. The OCR results are stored in hocr format. The hocr files are stored in the `output_hocr_file_dir`.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `images_dir_path` (str): Path to the directory where the images will be stored temporarily.
    - `output_hocr_file_dir` (str): Path to the hocr files.

    Docs:
    - https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
    """
    # ClearTextBack/temp/images/* should exist.
    os.system(f"rm -rf {images_dir_path}/*")

    image_file_paths = convert_pdf_to_images(pdf_file_path, images_dir_path)
    hocr_files = []

    for image_file_name in image_file_paths:
        hocr_file = image_file_name.replace(".jpeg", "")
        os.system(f"tesseract {image_file_name} {hocr_file} -l eng hocr")
        hocr_files.append(hocr_file)

    if not os.path.exists(output_hocr_file_dir):
        os.makedirs(output_hocr_file_dir)
    else:
        os.system(f"rm -rf {output_hocr_file_dir}/*")
    
    for hocr_file in hocr_files:
        with open(hocr_file + '.hocr', 'r') as f:
            hocr_content = f.read()
            hocr_content = add_styles_to_hocr(hocr_content)
        with open(f"{output_hocr_file_dir}/{os.path.basename(hocr_file)}.html", 'w') as f:
            f.write(hocr_content)
    
    os.system(f"rm -rf {images_dir_path}/*")


def pdf_to_hocr_images_tesseract(pdf_file_path: str, images_dir_path: str, output_images_dir: str, bbox_attribute: str = 'ocrx_word') -> None:
    """
    Converts PDF pages to images, performs OCR using Tesseract to obtain bounding boxes, 
    and plots these boxes on the images. This is useful for visualizing the OCR output using Tesseract.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `images_dir_path` (str): Directory to save intermediate images for OCR.
    - `output_images_dir` (str): Directory to save output images with bounding boxes plotted.
    - `bbox_attribute` (str): Attribute to use for bounding boxes. Default is `ocrx_word`.
        - You can choose between the following for this.
            - `ocr_line`
            - `ocrx_word`
            - `ocr_par`
        - The bounding box will be drawn around the text elements with the chosen attribute.
    """
    import pytesseract
    from PIL import Image, ImageDraw
    from pdf2image import convert_from_path
    from bs4 import BeautifulSoup

    assert bbox_attribute in ['ocr_line', 'ocrx_word', 'ocr_par'], "Invalid bbox_attribute. Choose between 'ocr_line', 'ocrx_word' and 'ocr_par'."

    images = convert_from_path(pdf_file_path)

    if not os.path.exists(images_dir_path):
        os.makedirs(images_dir_path)
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    for i, image in enumerate(images):
        if bbox_attribute == 'ocr_line' or bbox_attribute == 'ocrx_word':
            image_path = f"{images_dir_path}/page_{i+1}.png"
            image.save(image_path, 'PNG')
            
            hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
            soup = BeautifulSoup(hocr_output, 'html.parser')
            ocr_boxes = soup.find_all('span', class_=bbox_attribute)

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            for box in ocr_boxes:
                title = box['title']
                coords = title.split(';')[0].split(' ')
                x1, y1, x2, y2 = map(int, [coords[1], coords[2], coords[3], coords[4]])
                draw.rectangle([x1, y1, x2, y2], outline='red')
            
            output_image_path = f"{output_images_dir}/page_{i+1}_boxed.png".replace('//', '/')
            img.save(output_image_path)
        elif bbox_attribute == 'ocr_par':
            image_path = f"{images_dir_path}/page_{i+1}.png"
            image.save(image_path, 'PNG')
            
            hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
            soup = BeautifulSoup(hocr_output, 'html.parser')
            ocr_boxes = soup.find_all('p', class_=bbox_attribute)

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            for box in ocr_boxes:
                title = box['title']
                coords = title.split(';')[0].split(' ')
                x1, y1, x2, y2 = map(int, [coords[1], coords[2], coords[3], coords[4]])
                draw.rectangle([x1, y1, x2, y2], outline='red')
            
            output_image_path = f"{output_images_dir}/page_{i+1}_boxed.png".replace('//', '/')
            img.save(output_image_path)
    
    os.system(f"rm -rf {images_dir_path}/*")


def convert_docx_to_pdf(docx_path: str, pdf_path: str) -> str:
    """
    Convert a DOCX file to a PDF file.

    Args:
        `docx_path` (str): Path to the DOCX file.
        `pdf_path` (str): Path to save the PDF file.
    """
    from fpdf import FPDF
    from docx import Document

    doc = Document(docx_path)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for para in doc.paragraphs:
        pdf.cell(200, 10, txt=para.text, ln=True)
    pdf.output(pdf_path)
    return pdf_path


def convert_doc_to_pdf(doc_path: str, pdf_path: str) -> str:
    """
    Convert a DOC file to a PDF file.

    Args:
        `doc_path` (str): Path to the DOC file.
        `pdf_path` (str): Path to save the PDF file.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Welcome to Reportlab!")
    c.save()
    return pdf_path


def convert_pptx_to_pdf(pptx_path, pdf_path):
    """
    Convert a PPTX file to a PDF file, trying to maintain slide layouts and content. The conversion also contains a watermark.

    Args:
        `pptx_path` (str): Path to the PPTX file.
        `pdf_path` (str): Path to save the PDF file.
    """
    from spire.presentation import Presentation, FileFormat

    presentation = Presentation()
    presentation.LoadFromFile(pptx_path)
    presentation.SaveToFile(pdf_path, FileFormat.PDF)
    presentation.Dispose()
    return pdf_path


def split_pdf_pages(pdf_file_path: str, output_dir: str) -> list:
    """
    Splits a PDF file into individual pages. The split PDF pages are stored in the `output_dir`. The function returns a list of paths to the split PDF files.
    
    Args:
    - pdf_file_path (str): Path to the original PDF file.
    - output_dir (str): Directory to store the split PDF pages.
    """
    from PyPDF2 import PdfReader, PdfWriter

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf = PdfReader(pdf_file_path)
    page_files = []

    for i in range(len(pdf.pages)):
        output_pdf_path = f"{output_dir}/{os.path.splitext(os.path.basename(pdf_file_path))[0]}_xpage_{i+1}.pdf"
        writer = PdfWriter()
        writer.add_page(pdf.pages[i])
        with open(output_pdf_path, "wb") as f:
            writer.write(f)
        page_files.append(output_pdf_path)
        writer.close()

    return page_files


def extract_ocr_data_from_pdf(pdf_file_path: str, temp_image_dir: str, output_json_path: str = '', target: str = 'par', verbose: bool = True) -> list:
    """
    Extracts words, their bounding boxes, line IDs, paragraph IDs from a PDF using Tesseract OCR. The OCR data is stored as a JSON file if `output_json_path` is provided. The function returns the OCR data.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `temp_image_dir` (str): Temporary directory to store images for OCR processing.
    - `output_json_path` (str): Path to save the OCR data as a JSON file. Default is ''.
    - `target` (str): Target to extract the OCR data. Default is `word`.
        - You can choose between the following for this.
            - `word`: Extracts the `word` and its coresponding `bbox`, `line_id`, and `par_id`.
            - `line`: Extracts the `line` and its coresponding `bbox`, `line_id`, and `par_id`.
            - `par`: Extracts the `paragraph` and its coresponding `bbox`, `line_id`, and `par_id`.
    - `verbose` (bool): Flag to print the progress. Default is `True`.
    """
    import pytesseract
    from pdf2image import convert_from_path
    from bs4 import BeautifulSoup
    from tqdm.auto import tqdm
    
    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    assert target in ['word', 'line', 'par'], "Invalid target value. Choose between 'word', 'line' and 'par'."
    
    images = convert_from_path(pdf_file_path)
    if verbose:
        print(">> PDF converted to images successfully.")

    ocr_results = []

    if target == 'word':
        if verbose:
            print(">> Extracting words from the pages...")
        for i, image in tqdm(enumerate(images), desc="Processing pages"):
            image_path = f"{temp_image_dir}/page_{i+1}.png"
            image.save(image_path, 'PNG')

            hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
            soup = BeautifulSoup(hocr_output, 'html.parser')

            words = soup.find_all('span', class_='ocrx_word')

            for word in tqdm(words, desc=f"Processing words in page {i+1}"):
                text = word.get_text(strip=True)
                if text:
                    bbox_info = word['title'].split(';')
                    coords = bbox_info[0].split()[1:]
                    bbox = list(map(int, coords))
                    
                    line_id = word.find_previous('span', class_='ocr_line')['id'] if word.find_previous('span', class_='ocr_line') else None
                    par_id = word.find_previous('p', class_='ocr_par')['id'] if word.find_previous('p', class_='ocr_par') else None

                    ocr_results.append({
                        'bbox': {
                            'left': bbox[0], 
                            'top': bbox[1], 
                            'right': bbox[2], 
                            'bottom': bbox[3]
                        },
                        'line_id': line_id,
                        'par_id': par_id,
                        'text': text
                    })
    if target == 'line':
        if verbose:
            print(">> Extracting lines from the pages...")
        for i, image in tqdm(enumerate(images), desc="Processing pages"):
            image_path = f"{temp_image_dir}/page_{i+1}.png"
            image.save(image_path, 'PNG')

            hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
            soup = BeautifulSoup(hocr_output, 'html.parser')

            lines = soup.find_all('span', class_='ocr_line')

            for line in tqdm(lines, desc=f"Processing lines in page {i+1}"):
                text = ' '.join([span.get_text(strip=True) for span in line.find_all('span')])
                if text:
                    bbox_info = line['title'].split(';')
                    coords = bbox_info[0].split()[1:]
                    bbox = list(map(int, coords))
                    
                    par_id = line.find_previous('p', class_='ocr_par')['id'] if line.find_previous('p', class_='ocr_par') else None

                    ocr_results.append({
                        'bbox': {
                            'left': bbox[0], 
                            'top': bbox[1], 
                            'right': bbox[2], 
                            'bottom': bbox[3]
                        },
                        'line_id': line['id'],
                        'par_id': par_id,
                        'text': text
                    })
    if target == 'par':
        if verbose:
            print(">> Extracting paragraphs from the pages...")
        for i, image in tqdm(enumerate(images), desc="Processing pages"):
            image_path = f"{temp_image_dir}/page_{i+1}.png"
            image.save(image_path, 'PNG')

            hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
            soup = BeautifulSoup(hocr_output, 'html.parser')

            pars = soup.find_all('p', class_='ocr_par')

            for par in tqdm(pars, desc=f"Processing paragraphs in page {i+1}"):
                text = ' '.join([span.get_text(strip=True) for span in par.find_all('span') if span['class'][0] == 'ocrx_word'])
                if text.strip():
                    bbox_info = par['title'].split(';')
                    coords = bbox_info[0].split()[1:]
                    bbox = list(map(int, coords))
                    
                    ocr_results.append({
                        'bbox': {
                            'left': bbox[0], 
                            'top': bbox[1], 
                            'right': bbox[2], 
                            'bottom': bbox[3]
                        },
                        'line_id': None,
                        'par_id': par['id'],
                        'text': text
                    })

    if verbose:
        print(">> OCR data extracted successfully.")

    os.system(f"rm -rf {temp_image_dir}/*")
    if verbose:
        print(">> Temporary images deleted successfully.")

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(ocr_results, f, indent=4)
        if verbose:
            print(f">> OCR data saved at: {output_json_path}")
    else:
        if verbose:
            print(">> OCR data not saved.")
    return ocr_results


def tag_pdf(pdf_file_path: str, output_pdf_path: str, title: str = 'Document Title', lang: str = 'en-US') -> str:
    """
    Makes a PDF file accessible. Adds tags to the PDF file to make it accessible. The output PDF file is saved at `output_pdf_path`.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `output_pdf_path` (str): Path to the output PDF file.
    - `title` (str): Title of the document.
    - `lang` (str): Language of the document.
    """
    from pdfixsdk.Pdfix import GetPdfix, PdfAccessibleParams, kSaveFull

    pdfix  = GetPdfix()
    if pdfix is None:
        raise Exception('Pdfix Initialization fail')

    doc = pdfix.OpenDoc(pdf_file_path, '')
    if doc is None:
        raise Exception('Unable to open pdf')

    tmpl = doc.GetTemplate()
    for i in range(0, doc.GetNumPages()):
        tmpl.AddPage(i, 0, None)
    tmpl.Update(0, None)

    accessibleParams = PdfAccessibleParams()
    if not doc.MakeAccessible(accessibleParams, title, lang, 0, None):
        raise Exception(pdfix.GetError())
    
    if not doc.Save(output_pdf_path, kSaveFull):
        raise Exception(pdfix.GetError())

    doc.Close()
    return output_pdf_path


def extract_and_save_pdf_tags(pdf_file_path: str, output_file_dir: str) -> typing.Tuple[str, str]:
    """
    Extracts the tags from the PDF file and saves them as an HTML file. The function returns the path to the saved tags HTML file and the path to the tagged PDF file.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `output_file_dir` (str): Directory to save the tags HTML file.
    """
    import re
    from bs4 import BeautifulSoup
    from python.detect_tag import convert_pdf

    def clean_tags(tag_string):
        cleaned_string = tag_string.decode("utf-8") # Convert bytes to string
        cleaned_string = re.sub(r"b'([^']*)'", r"\1", cleaned_string) # Remove b' and ' from the string
        cleaned_string = cleaned_string.replace("'layout'", "layout") # Remove ' from layout
        cleaned_string = re.sub(r"\"\"=\"", "=\"", cleaned_string) # Remove empty attributes
        cleaned_string = re.sub(r"\'([^']*)\'", r"\1", cleaned_string) # Remove ' from the string
        return cleaned_string

    _, original_tags = convert_pdf(pdf_file_path)
    
    temp_pdf_file_path = os.path.join(output_file_dir, os.path.basename(pdf_file_path))
    tagged_pdf_path = tag_pdf(pdf_file_path, temp_pdf_file_path)
    _, new_tags = convert_pdf(tagged_pdf_path)
    os.remove(tagged_pdf_path)

    soup_original = BeautifulSoup(original_tags, 'html.parser')
    soup_new = BeautifulSoup(new_tags, 'html.parser')
    
    original_tags = [str(tag) for tag in soup_original.find_all()[1:]]
    new_tags = [tag for tag in soup_new.find_all()[1:]]
    
    for tag in new_tags:
        if str(tag) not in original_tags:
            tag['generated_tag'] = 'true'
    
    cleaned_tags = clean_tags(soup_new)

    output_file_path = os.path.join(output_file_dir, os.path.basename(tagged_pdf_path).replace(".pdf", "_tags.html"))
    
    with open(output_file_path, "w") as file:
        file.write(cleaned_tags)

    return output_file_path, tagged_pdf_path


def parse_text(text: str) -> str:
    """
    Handle some of the unicode and other escape sequence issues.

    Args:
    - `text` (str): Text to be parsed.
    """
    import unidecode

    parsed_text = unidecode.unidecode(text)
    return parsed_text


def image_to_pdf(image_file_path: str, output_pdf_path: str) -> None:
    """
    Converts an image file to a PDF file using Pillow.

    Args:
    - `image_file_path` (str): Path to the image file.
    - `output_pdf_path` (str): Path to save the PDF file.
    """
    from PIL import Image
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4

    page_size = A4

    with Image.open(image_file_path) as img:
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height

        if img_width > img_height:
            page_width, page_height = max(page_size), min(page_size)
        else:
            page_width, page_height = min(page_size), max(page_size)

        max_width = page_width * 0.95 
        max_height = page_height * 0.95  

        if (max_width / max_height > aspect_ratio):
            max_width = max_height * aspect_ratio
        else:
            max_height = max_width / aspect_ratio

        c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))
        img_x = (page_width - max_width) / 2
        img_y = (page_height - max_height) / 2

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save("temp_image_for_pdf.jpg", "JPEG", quality=95)

        c.drawInlineImage("temp_image_for_pdf.jpg", img_x, img_y, width=max_width, height=max_height)
        c.showPage()
        c.save()
        os.remove("temp_image_for_pdf.jpg") 


def extract_ocr_data_with_tags_from_pdf(pdf_file_path: str, output_json_path: str = '', target: str = 'par', verbose: bool = True) -> list:
    """
    Extracts OCR data from PDF and augments it with HTML tag and style information. The OCR data is stored as a JSON file if `output_json_path` is provided. The function returns the OCR data.

    Args:
    - `pdf_file_path` (str): Path to the PDF file.
    - `output_json_path` (str): Path to save the OCR data as a JSON file. Default is ''.
    - `target` (str): Target to extract the OCR data. Default is `par`.
        - You can choose between the following for this.
            - `line`: Extracts the `line` and its coresponding `bbox`, `tag`, and `text`.
            - `par`: Extracts the `paragraph` and its coresponding `bbox`, `tag`, and `text`.
    - `verbose` (bool): Flag to print the progress. Default is `True`.
    """
    import json, os, pytesseract, difflib
    from pdf2image import convert_from_path
    from bs4 import BeautifulSoup

    assert target in ['line', 'par'], "Invalid target value. Choose between 'line' and 'par'."

    temp_image_dir = TEMP_IMAGE_DIR
    html_output_path = HTML_TEMP_DIR
    tags_output_dir = TAGS_TEMP_DIR

    tags_output_path, _ = extract_and_save_pdf_tags(pdf_file_path, os.path.dirname(tags_output_dir))

    images = convert_from_path(pdf_file_path)
    if verbose:
        print(">> PDF converted to images successfully.")

    ocr_results = []

    with open(tags_output_path, 'r') as file:
        tags_content = file.read()
    tag_soup = BeautifulSoup(tags_content, 'html.parser')

    for i, image in enumerate(images):
        image_path = f"{temp_image_dir}/page_{i+1}.png"
        image.save(image_path, 'PNG')

        hocr_output = pytesseract.image_to_pdf_or_hocr(image_path, extension='hocr')
        soup = BeautifulSoup(hocr_output, 'html.parser')

        mapper = {'line': 'ocr_line', 'par': 'ocr_par'}

        if target == 'line':
            elements = soup.find_all('span', class_ = mapper[target])
            for element in elements:
                text = ' '.join([span.get_text(strip=True) for span in element.find_all('span')])
                if text:
                    bbox_info = element['title'].split(';')
                    coords = bbox_info[0].split()[1:]
                    bbox = list(map(int, coords))

                    max_ratio = 0
                    selected_tag = None
                    for tag in tag_soup.find_all(True):
                        tag_text = tag.get_text(strip=True)
                        tag_text = ' '.join(tag_text.split())

                        ratio = difflib.SequenceMatcher(None, text, tag_text).ratio()
                        if ratio > max_ratio:
                            max_ratio = ratio
                            selected_tag = tag
                    tag_name = selected_tag.name if selected_tag else 'No tag'

                    text = parse_text(text)

                    ocr_results.append({
                        'bbox': {'left': bbox[0], 'top': bbox[1], 'right': bbox[2], 'bottom': bbox[3]},
                        'text': text,
                        'tag': tag_name,
                    })
        if target == 'par':
            elements = soup.find_all('p', class_ = mapper[target])
            for element in elements:
                text = ' '.join([span.get_text(strip=True) for span in element.find_all('span') if span['class'][0] == 'ocrx_word'])
                if text:
                    bbox_info = element['title'].split(';')
                    coords = bbox_info[0].split()[1:]
                    bbox = list(map(int, coords))

                    max_ratio = 0
                    selected_tag = None
                    for tag in tag_soup.find_all(True):
                        tag_text = tag.get_text(strip=True)
                        tag_text = ' '.join(tag_text.split())  
                        
                        ratio = difflib.SequenceMatcher(None, text, tag_text).ratio()
                        if ratio > max_ratio:
                            max_ratio = ratio
                            selected_tag = tag
                    tag_name = selected_tag.name if selected_tag else 'No tag'

                    text = parse_text(text)

                    ocr_results.append({
                        'bbox': {'left': bbox[0], 'top': bbox[1], 'right': bbox[2], 'bottom': bbox[3]},
                        'text': text,
                        'tag': tag_name,
                    })
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(ocr_results, f, indent=4)
        if verbose:
            print(f">> OCR data saved at: {output_json_path}")

    os.system(f"rm -rf {temp_image_dir}/*")
    os.system(f"rm -rf {tags_output_dir}/*")
    os.system(f"rm -rf {html_output_path}")
    if verbose:
        print(">> Temporary files cleaned up.")

    return ocr_results


if __name__ == "__main__":
    """
    >>> python3 -m tools.utils
    """
    pdf_file_path = "pdf/grsupra_ebrochure.pdf"
    image_file_path = "pdf/000357_page_2.png"
    output_pdf_path = "pdf/temp.pdf"
    
    extract_ocr_data_with_tags_from_pdf(pdf_file_path, output_json_path="temp/ocr_data.json", target='par', verbose=True)