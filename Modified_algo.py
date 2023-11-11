import cv2
import streamlit as st
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image
from reportlab.lib.utils import ImageReader
import numpy as np
from difflib import SequenceMatcher
import pytesseract
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\HP\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"


def extract_frames(video_path, frame_rate, threshold):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Set the desired frame rate (in frames per second)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
    print('Frames Per second', fps)

    print('Frames Extraction Started...')
    n = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if (frame_rate * n) % fps == 0:
            is_duplicate = False
            for existing_frame in frames:
                if is_similar(frame, existing_frame, threshold):
                    is_duplicate = True
                    break
            if not is_duplicate:
                cv2.imwrite(f'frames/img_{i}.jpg', frame)
                frames.append(frame)
                i += 1
        n += 1
        if not ret:
            break

    cap.release()
    print('Frames Extraction Done!')
    print('Total Frames:', len(frames))
    return frames


def extract_frames_ocr(frames):
    ocr_frames = [frames[0]]

    for i in tqdm(range(len(frames) - 1)):

        sent1, detect1 = combine_sentences(frames[i])
        sent2, detect2 = combine_sentences(frames[i + 1])
        similar_sent = similar_sentences(sent1, sent2)
        if len(similar_sent) == 0:
            ocr_frames.append(frames[i + 1])
            continue
        final_coordinates, _ = final_coord(similar_sent, detect1, detect2)
        if len(final_coordinates) == 0 or len(final_coordinates) == 1:
            ocr_frames.append(frames[i + 1])
            continue
        else:
            result_frame = ocr_masking(frames[i + 1], final_coordinates)
            ocr_frames.append(result_frame)

    print('ocr frames: ', len(ocr_frames))
    return ocr_frames


def is_similar(frame1, frame2, threshold=0.9):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hist_frame1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
    hist_frame2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

    hist_frame1 /= hist_frame1.sum()
    hist_frame2 /= hist_frame2.sum()

    intersection = cv2.compareHist(hist_frame1, hist_frame2, cv2.HISTCMP_INTERSECT)
    return intersection >= threshold


def combine_sentences(frame):
    sentences = []  # To store sentences
    current_sentence = ""  # To accumulate text within a sentence
    detections = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)

    for i in range(len(detections['text'])):
        text = detections['text'][i]
        if text.strip():  # Check if the text is not empty or whitespace
            if '.' not in text or '..' in text:  # Check if the text contains word characters
                current_sentence += ' ' + text  # Reset the current sentence
            elif '.' in text or '..' not in text:
                current_sentence += ' ' + text
                sentences.append({'sentence': current_sentence})
                current_sentence = ''
    return sentences, detections


def similar_sentences(sentences1, sentences2, similarity_threshold=0.95):
    similar_sent = []
    for sentence2 in sentences2:
        for sentence1 in sentences1:
            if SequenceMatcher(None, sentence2['sentence'], sentence1['sentence']).ratio() > similarity_threshold:
                similar_sent.append(sentence2['sentence'])
    return similar_sent


def final_coord(similar_sent, detections1, detections2):
    final_coordinates = []
    final_sentence = similar_sent[0].split()[:2] + similar_sent[-1].split()[-2:]

    for i in range(len(detections1['text']) - 1):
        if detections1['text'][i] == final_sentence[0] and detections1['text'][i + 1] == final_sentence[1]:

            start_index = detections2['text'].index(final_sentence[0])
            start_coord = (
                detections2['left'][start_index], detections2['top'][start_index], detections2['width'][start_index],
                detections2['height'][start_index])
            final_coordinates.append(start_coord)
        elif detections1['text'][i] == final_sentence[-2] and detections1['text'][i + 1] == final_sentence[-1]:

            end_index = detections2['text'].index(final_sentence[-1])
            end_coord = (detections2['left'][end_index], detections2['top'][end_index], detections2['width'][end_index],
                         detections2['height'][end_index])
            final_coordinates.append(end_coord)

    return final_coordinates, final_sentence


def ocr_masking(frame, final_coord):
    (x1, y1, x2, y2) = final_coord[0][0] - 500, final_coord[0][0] - 50, final_coord[1][0] + final_coord[1][2] + 800, \
                       final_coord[1][1] + final_coord[1][3] + 50
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    mask = np.ones(frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 0
    result_image = cv2.bitwise_and(frame, frame, mask=mask)

    return result_image


def frames_to_pdf(frames, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=(frames[0].shape[1], frames[0].shape[0]))

    print('Frames to PDF Started...')
    for idx, frame in enumerate(frames):
        img_buffer = BytesIO()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
        frame_pil.save(img_buffer, format='JPEG')

        img_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        img_reader = ImageReader(img_buffer)

        c.drawImage(img_reader, 0, 0, width=frame.shape[1], height=frame.shape[0])
        c.showPage()

    c.save()
    print('PDF with frames created successfully!')


def main():
    st.title("Video to PDF Converter")
    video_path = None
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])
    if video_file:
        video_path = f"custom_video.{video_file.type.split('/')[1]}"
        with open(video_path, "wb") as f:
            f.write(video_file.read())

    output_pdf = st.text_input("Enter the path for the output PDF file:",'Output.pdf')
    if st.button("Convert Video to PDF"):
        frames = extract_frames(video_path, frame_rate=1, threshold=0.89)
        ocr_frames = extract_frames_ocr(frames)
        frames_to_pdf(ocr_frames, output_pdf)


if __name__ == "__main__":
    main()
