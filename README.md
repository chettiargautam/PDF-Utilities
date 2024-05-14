# PDF Processing Utility Functions

This Python module provides various utility functions for PDF processing, including conversion, tagging, OCR, and more (This is only for educational purposes and the code here is for viewing only, please do not use for commercial distribution). From converting PDFs to images to extracting OCR data, these functions aim to streamline various tasks involved in PDF manipulation and analysis.


## Environment Setup

You can install the required dependencies using pip or conda:

```bash
pip install -r requirements.txt
```

```bash
conda install --file requirements.txt
```


## Dependency and Module Installation

You can install the required dependencies using pip or conda:

```bash
pip install -r requirements.txt
```

```bash
conda install --file requirements.txt
```


## Function Details

1. **`convert_pdf_to_images(pdf_file_path: str, output_save_dir: str = '') -> list`**

   - This function converts a PDF file to a list of images.
   - It utilizes the `pdf2image` library to perform the conversion.
   - Each page of the PDF is converted to a separate image file.
   - The images are stored in the specified `output_save_dir`.
   - Returns a list of paths to the generated image files.

2. **`pdf_to_html_pdf2txt(pdf_file_path: str, output_file_path: str, redact_anchor_tags: bool = True, visualize_bboxes: bool = True) -> None`**

   - Converts a PDF file to an HTML file using the `pdf2txt` tool.
   - The resulting HTML file can be redacted to remove anchor tags and visualized to show bounding boxes.
   - Redacting anchor tags is useful for removing hyperlinks from the text.
   - Visualizing bounding boxes aids in understanding the layout of text in the PDF.
   - Saves the HTML output to the specified `output_file_path`.

3. **`get_image_size(image_path : str) -> tuple`**

   - Retrieves the dimensions (width and height) of an image.
   - Utilizes the `Pillow` library to open and process the image file.
   - Returns a tuple containing the width and height of the image.

4. **`add_styles_to_hocr(hocr_content : str) -> str`**

   - Adds CSS styles to the hOCR (HTML-based OCR) content.
   - Useful for visualizing bounding boxes around recognized text elements.
   - Uses `BeautifulSoup` to parse and manipulate the hOCR content.
   - Returns the modified hOCR content with added CSS styles.

5. **`pdf_to_hocr_html_tesseract(pdf_file_path: list, images_dir_path: str, output_hocr_file_dir: str) -> None`**

   - Performs OCR on images extracted from a PDF using Tesseract.
   - Converts PDF pages to images, performs OCR, and saves the OCR results in hOCR format.
   - Each image's OCR output is stored as an HTML file in the specified directory.
   - Requires the `pdf2image` and `tesseract` libraries for PDF to image conversion and OCR, respectively.
   - Removes temporary image files after processing.

6. **`pdf_to_hocr_images_tesseract(pdf_file_path: str, images_dir_path: str, output_images_dir: str, bbox_attribute: str = 'ocrx_word') -> None`**

   - Converts PDF pages to images, performs OCR using Tesseract, and plots bounding boxes on the images.
   - Useful for visualizing OCR output with bounding boxes.
   - Supports different bounding box attributes such as word, line, and paragraph.
   - Requires `pytesseract` for OCR and `PIL` for image manipulation.

7. **`convert_docx_to_pdf(docx_path: str, pdf_path: str) -> str`**

   - Converts a DOCX file to a PDF file.
   - Utilizes the `FPDF` library to create a PDF and `python-docx` to extract text from the DOCX file.
   - Returns the path to the generated PDF file.

8. **`convert_doc_to_pdf(doc_path: str, pdf_path: str) -> str`**

   - Converts a DOC file to a PDF file.
   - Uses the `reportlab` library to generate a PDF with custom content.
   - Returns the path to the generated PDF file.

9. **`convert_pptx_to_pdf(pptx_path, pdf_path) -> str`**

   - Converts a PPTX file to a PDF file while maintaining slide layouts and content.
   - Adds a watermark to the converted PDF.
   - Relies on the `spire.presentation` library for conversion.
   - Returns the path to the generated PDF file.

10. **`split_pdf_pages(pdf_file_path: str, output_dir: str) -> list`**

    - Splits a PDF file into individual pages.
    - Saves each page as a separate PDF file in the specified output directory.
    - Uses `PyPDF2` library for PDF manipulation.
    - Returns a list of paths to the split PDF files.

11. **`extract_ocr_data_from_pdf(pdf_file_path: str, temp_image_dir: str, output_json_path: str = '', target: str = 'par', verbose: bool = True) -> list`**

    - Extracts OCR data (words, bounding boxes, line IDs, paragraph IDs) from a PDF using Tesseract OCR.
    - OCR results are stored as a JSON file if `output_json_path` is provided.
    - Supports extraction at word, line, or paragraph level (specified by `target`).
    - Requires `pytesseract` and `pdf2image` libraries for OCR and PDF to image conversion, respectively.

12. **`tag_pdf(pdf_file_path: str, output_pdf_path: str, title: str = 'Document Title', lang: str = 'en-US') -> str`**

    - Makes a PDF file accessible by adding tags to enhance accessibility.
    - Uses `Pdfix` library for PDF manipulation.
    - Tags improve the reading order and navigation of the document for users with disabilities.
    - Returns the path to the tagged PDF file.

13. **`extract_and_save_pdf_tags(pdf_file_path: str, output_html_path: str) -> None`**

    - Extracts tags from a PDF file and saves them as an HTML file.
    - Useful for visualizing the document structure and tags.
    - Requires `Pdfix` library for PDF manipulation.

14. **`parse_text(text: str) -> str`**

    - Handles Unicode and other escape sequence issues in the input text.
    - Replaces non-printable characters and escapes sequences with their corresponding Unicode representations.
    - Returns the processed text.

15. **`image_to_pdf(image_path: str, pdf_path: str) -> None`**

    - Converts an image file to a PDF file.
    - Uses the `reportlab` library to generate a PDF with the image.
    - Supports various image formats (JPEG, PNG, etc.).
    - Saves the generated PDF to the specified path.

16. **`extract_ocr_data_with_tags_from_pdf(pdf_file_path: str, output_json_path: str = '', target: str = 'par', verbose: bool = True) -> list`**

    - Extracts OCR data from a PDF and augments it with HTML tag and style information.
    - Supports extraction at word, line, or paragraph level (specified by `target`).
    - Generates a JSON file containing the augmented OCR data if `output_json_path` is provided.
    - Requires `pdf2image` and `pytesseract` libraries for PDF to image conversion and OCR

, respectively.


## Sample Usage

```python
# Sample code for extracting OCR data with tags from a PDF
pdf_file_path = "pdf/example.pdf"
extract_ocr_data_with_tags_from_pdf(pdf_file_path, output_json_path="output/ocr_data.json", target='par', verbose=True)
```


## Citation

If you find this library useful for your work, please consider citing it as follows:

```
[Gautam Chettiar]. PDF Processing Utility Functions. 2024, https://github.com/chettiargautam/PDF-Utilities
```