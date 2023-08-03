import numpy as np
import cv2
import pytesseract
import easyocr
import streamlit
from PIL import Image
from pytesseract import Output
import PyPDF2
import io
from transformers import DetrFeatureExtractor
import torch


# OCR READER
reader = easyocr.Reader(['en'])
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karnb\tesseract-ocr-w64-setup-5.3.1.20230401.exe"


def ocr(image):
    # image=cv2.imread(image)
    image = np.array(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(img_gray)
    return text


def DLA(image):
    image = np.array(image)
    result = reader.readtext(image,detail=0)
    return result


# CONVERTING PDF TO IMAGE

# def convert_pdf_to_images(uploaded_pdf):
#     path=os.path.abspath(uploaded_pdf)
#     images = []
#     pdf_file = Image.open(path)
#     for page_num in range(pdf_file.n_frames):
#         pdf_file.seek(page_num)
#         image = pdf_file.copy()
#         images.append(image)
#     return images

def draw_boxes(img):
    img=np.array(img)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

def pdf_to_png(pdf_file):
    images = []
    pdf = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf.pages)

    for page_num in range(num_pages):
        page = pdf.pages[page_num]
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(page)

        pdf_bytes = io.BytesIO()
        pdf_writer.write(pdf_bytes)
        pdf_bytes.seek(0)

        img = Image.open(pdf_bytes)
        images.append(img)

    return images


import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    streamlit.pyplot(plt)
    # plt.show()


# MICROSOFT TABLE TRANSFORMER MODEL
feature_extractor = DetrFeatureExtractor()
from transformers import TableTransformerForObjectDetection
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

def table_detection(image):
    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()
    with torch.no_grad():
        outputs = model(**encoding)
    width, height = image.size
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[(height, width)])[0]
    return results

