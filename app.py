import pytesseract
import streamlit as st
import helper
from helper import ocr, DLA, pdf_to_png
from PIL import Image

st.title("DocAI App: Text Extraction,Layout and Table Detection")

uploaded_file = st.file_uploader("Upload a document (PNG/PDF)", type=['png', 'pdf'])

if uploaded_file is not None:

    # Check if the uploaded file is a PDF
    if uploaded_file.type=='application/pdf':
        # Convert the PDF to images and process each page
        images = pdf_to_png(uploaded_file)
        for i, image in enumerate(images):
            st.subheader(f"Page {i + 1}")
            st.image(image, caption=f"Page {i + 1}", use_column_width=True)

            # Perform OCR using Tesseract
            text = ocr(image)
            st.write("Extracted Text:")
            c=st.container
            c.write(text)

            # Perform layout analysis using EasyOCR
            layout_result = DLA(image)
            st.write("Layout Analysis:")
            c2=st.container
            c2.write(layout_result)

    else:
        # For PNG files, process the single image
        image = Image.open(uploaded_file)
        st.subheader("Uploaded image")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("Bounding Boxes")
        bbimg=helper.draw_boxes(image)
        st.image(bbimg)

        # Perform OCR using Tesseract
        text = ocr(image)
        st.subheader("Extracted Text:")
        st.write(text)

        # Table extraction
        st.subheader("Table Extraction")
        result=helper.table_detection(image)
        helper.plot_results(image, result['scores'], result['labels'], result['boxes'])

        # Perform layout analysis using EasyOCR
        layout_result = DLA(image)
        st.subheader("Layout Analysis:")
        st.write(layout_result)


