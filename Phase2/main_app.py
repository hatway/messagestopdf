from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
from PIL import ImageGrab
from kivy.uix.popup import Popup
from kivy.clock import Clock
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image
from kivy.animation import Animation
from kivy.properties import ObjectProperty
from kivymd.uix.screen import MDScreen
import numpy as ny
import shutil
from win32api import GetSystemMetrics
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import time

Window.size = (500, 600)

Builder.load_file('record.kv')  # Load the Screenrecordmenu Kivy design
Builder.load_file('editor.kv')
Builder.load_file('videoplayer.kv')


class Screenrecordmenu(MDScreen):

    def screenrecord(self):
        MDApp.get_running_app().root_window.minimize()
        min = time.time() + 60 * eval(self.ids.screenrecordmin.text)
        height = GetSystemMetrics(1)
        width = GetSystemMetrics(0)

        file_name = f'{self.ids.screenrecordname.text}.mp4'
        format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        final_video = cv2.VideoWriter(file_name, format, 8, (width, height))

        while time.time() < min:
            img = ImageGrab.grab(bbox=(0, 0, width, height))
            img_ny = ny.array(img)
            final_image = cv2.cvtColor(img_ny, cv2.COLOR_BGR2RGB)
            final_video.write(final_image)

        MDApp.get_running_app().root_window.restore()

class Videoeditor(MDScreen):
    start_time_input = ObjectProperty(None)
    end_time_input = ObjectProperty(None)
    video_name_input = ObjectProperty(None)
    output_label = ObjectProperty(None)
    cut_button = ObjectProperty(None)

    def cut_video(self):
        start_time = float(self.start_time_input.text)
        end_time = float(self.end_time_input.text)
        input_file = self.video_name_input.text
        output_file = 'output.mp4'

        try:
            ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
            self.output_label.text = f"Video cut successfully from {start_time:.2f}s to {end_time:.2f}s."
        except Exception as e:
            self.output_label.text = f"Error: {str(e)}"

class CustomPopup(Popup):

    def __init__(self, frames, pdf_callback, **kwargs):
        super(CustomPopup, self).__init__(**kwargs)
        self.frames = frames
        self.pdf_callback = pdf_callback

    def display_message(self, message):
        message_label = self.ids.message_label
        message_label.text = message

        # Fade in animation
        animation_in = Animation(opacity=1, duration=0.5)
        animation_in.start(message_label)

        # Schedule a fade-out animation after a delay
        def fade_out_label(dt):
            animation_out = Animation(opacity=0, duration=0.5)
            animation_out.start(message_label)

        Clock.schedule_once(fade_out_label, 2)

    def save_popup(self):
        new_name = self.ids.new_name.text
        if new_name:
            output_pdf = f"{new_name}.pdf"
            self.pdf_callback(output_pdf)
            self.display_message(f"File '{output_pdf}' saved successfully!")
        else:
            self.display_message("Please enter a filename.")


class Mylayout(MDScreen):

    def extract_frames(self, video_path, frame_rate, threshold):
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
                    if self.is_similar(frame, existing_frame, threshold):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    frames.append(frame)
                    i += 1
            n += 1
            if not ret:
                break

        cap.release()
        print('Frames Extraction Done!')
        print('Total Frames:', len(frames))
        return frames

    def is_similar(self, frame1, frame2, threshold=0.9):
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        hist_frame1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
        hist_frame2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

        hist_frame1 /= hist_frame1.sum()
        hist_frame2 /= hist_frame2.sum()

        intersection = cv2.compareHist(hist_frame1, hist_frame2, cv2.HISTCMP_INTERSECT)
        return intersection >= threshold

    def frames_to_pdf(self, frames, output_pdf, callback):
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
        callback(output_pdf)  # Call the callback function with the PDF filename

    def extract_frames_and_create_pdf(self):
        # Call your frame extraction and PDF generation functions here
        video_path = 'video1.mp4'

        frames = self.extract_frames(video_path, frame_rate=1, threshold=0.89)  # Extract frames

        # Define a callback function to open the rename popup after PDF is created
        def open_rename_popup(output_pdf):
            custom_popup = CustomPopup(frames=frames, pdf_callback=self.rename_pdf)
            custom_popup.ids.new_name.text = output_pdf[:-4]  # Set the default filename
            custom_popup.open()

        # Call the frames_to_pdf function with the callback
        self.frames_to_pdf(frames, 'output_frames.pdf', open_rename_popup)

    def rename_pdf(self, new_name):
        # Implement the code to rename the PDF file here
        # You can use os.rename or shutil.move to rename the file
        old_pdf = 'output_frames.pdf'
        new_pdf = f'{new_name}'
        shutil.move(old_pdf, new_pdf)
        # Example: os.rename(old_pdf, new_pdf)
        print(f'Renamed "{old_pdf}" to "{new_pdf}"')
class Main(MDApp):
    def build(self):
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Teal'
        return Builder.load_file('main_app.kv')  # Load the modified Kivy design

if __name__ == '__main__':
    Main().run()
