from kivy.uix.boxlayout import BoxLayout
from kivy.uix.video import Video
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.popup import Popup
from kivy.animation import Animation
from kivy.uix.label import Label
from kivy.app import App
from kivy.clock import Clock  # Import the Clock module

import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from reportlab.lib.utils import ImageReader
from kivy.uix.textinput import TextInput  # Import TextInput
from kivy.uix.label import Label  # Import Label
import os

from kivy.core.window import Window

from stitching import Stitcher

# Set the background color for the app
Window.clearcolor = (0.8, 0.8, 0.8, 0.7)

# Set the theme for the pop-up
popup_theme = {
    'title': 'PDF Conversion',
    'title_align': 'center',
    'title_color': (1, 1, 1, 1),
    'background_color': (0.7, 0.7, 0.9, 0.9),
    'content': Label(text='Please wait...', color=(0.5, 1, 1, 1)),
    'size_hint': (None, None),
    'size': (400, 200),
}


class VideoPlayerApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical', spacing=20, padding=50)
        self.video = Video(source='video1.mp4', state='play', options={'eos': 'loop'}, size_hint=(1, 0.5))
        self.root.add_widget(self.video)

        self.custom_button = RoundedButton(text='Convert to Pdf', size_hint=(0.4, 0.1), pos_hint={'center_x': 0.5},
                                           conversion_function=self.video_to_pdf)
        self.root.add_widget(self.custom_button)

    def video_to_pdf(self):
        video_path = 'video1.mp4'
        output_pdf = 'output_frames5.pdf'

        # Show the customized pop-up
        self.conversion_popup = Popup(**popup_theme)
        self.conversion_popup.open()

        # Schedule the conversion function to run in the background
        Clock.schedule_once(lambda dt: self.perform_conversion(video_path, output_pdf), 0.1)

    def perform_conversion(self, video_path, output_pdf, stitching=True):
        if not stitching:
            frames = extract_frames(video_path, frame_rate=1, threshold=0.89)
            frames_to_pdf(frames, output_pdf)
        else:
            frames, _ = extract_frames_stitching(video_path)
            frames_to_pdf_stitching(frames, output_pdf)

        # Close the pop-up
        self.conversion_popup.dismiss()

        # Show "Conversion Complete" popup with renaming and file location options
        popup_content = BoxLayout(orientation='vertical')

        # Text input for renaming the PDF file
        rename_input = TextInput(hint_text='Rename File')
        popup_content.add_widget(rename_input)

        download_button = RoundedButton(text='Save', background_color=(60 / 255, 150 / 255, 150 / 255, 1),
                                        conversion_function=lambda: self.show_location_popup(output_pdf,
                                                                                             rename_input.text))
        popup_content.add_widget(download_button)

        # Create a BoxLayout to hold the label and button, and add it to popup_content
        label_and_button_layout = BoxLayout(orientation='vertical', spacing=10, padding=5)  # Adjust spacing and padding
        popup_content.add_widget(label_and_button_layout)

        # Label to display the file location
        self.location_label = Label()
        label_and_button_layout.add_widget(self.location_label)

        # Create a beautiful separator
        separator = Label(text='-' * 30, color=(0.2, 0.2, 0.2, 1))
        popup_content.add_widget(separator)

        # Set the popup size and add content
        popup = Popup(title='Conversion Completed', content=popup_content, size_hint=(None, None), size=(400, 200))

        # Customize the theme for the "Conversion Completed" pop-up
        popup.background_color = (0.7, 0.7, 0.9, 0.9)
        popup.title_color = (1, 1, 1, 1)

        # Open the popup
        popup.open()


    def show_location_popup(self, original_pdf_path, new_name):
        if new_name.strip():
            # If a new name is provided, rename the PDF file
            pdf_dir = os.path.dirname(original_pdf_path)
            new_pdf_path = os.path.join(pdf_dir, new_name + '.pdf')
            os.rename(original_pdf_path, new_pdf_path)
        else:
            # If no new name is provided, keep the original name
            new_pdf_path = original_pdf_path

        # Get the absolute path of the PDF file
        abs_pdf_path = os.path.abspath(new_pdf_path)

        # Create a new popup to display the file location
        location_popup = Popup(title='File Location', content=Label(text=f'File location: {abs_pdf_path}'),
                               size_hint=(None, None), size=(400, 200), auto_dismiss=True)

        # Customize the theme for the location popup
        location_popup.background_color = (0.7, 0.7, 0.9, 0.9)
        location_popup.title_color = (1, 1, 1, 1)

        # Show the location popup with a fading effect
        Animation(opacity=0, duration=2).start(location_popup)
        location_popup.open()
    # def rename_and_display_location(self, original_pdf_path, new_name):
    #     if new_name.strip():
    #         # If a new name is provided, rename the PDF file
    #         pdf_dir = os.path.dirname(original_pdf_path)
    #         new_pdf_path = os.path.join(pdf_dir, new_name + '.pdf')
    #         os.rename(original_pdf_path, new_pdf_path)
    #     else:
    #         # If no new name is provided, keep the original name
    #         new_pdf_path = original_pdf_path
    #
    #     # Get the absolute path of the Pdf file
    #     abs_pdf_path = os.path.abspath((new_pdf_path))
    #
    #     # Display the file location in the label
    #     self.location_label.text = f'File location: {abs_pdf_path}'


class RoundedButton(Button):
    def __init__(self, conversion_function, **kwargs):
        super(RoundedButton, self).__init__(**kwargs)
        self.background_normal = ''  # Clear the normal background
        self.background_color = (60 / 255, 150 / 255, 150 / 255, 1)
        self.conversion_function = conversion_function  # Function for video to PDF conversion

    def on_press(self):
        self.background_color = (0.5, 0.5, 0.5, 1)  # Change the color when pressed

    def on_release(self):
        self.background_color = (60 / 255, 150 / 255, 150 / 255, 1)  # Revert to normal color
        self.conversion_function()


def extract_frames(video_path, frame_rate, threshold):
    # Your extraction code here...
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
                frames.append(frame)
                i += 1
        n += 1
        if not ret:
            break

    cap.release()
    print('Frames Extraction Done!')
    print('Total Frames:', len(frames))
    return frames

def extract_frames_stitching(video_path, frame_rate=12):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Set the desired frame rate (in frames per second)
    # Information of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Frames Per second', fps)
    print('Total frame count', length)
    print('Frames Extraction Started...')
    seperate_count = 1
    if length / frame_rate > 35:
        seperate_count = int(length / frame_rate) + 1
    frame_order = 0
    first_frames = 0
    while True:
        ret, frame = cap.read()
        if (frame_order + 3) % frame_rate == 0 or frame_order == length - 1:
            frame = cv2.resize(frame, (480, 720),  cv2.INTER_AREA)
            if first_frames > 2:
                frame = frame[90:720, 0:480]
            else:
                first_frames += 1
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frames.append(frame)
        frame_order += 1
        if not ret:
            break
    cap.release()
    print('Frames Extraction Done!')
    print('Total Frames:', len(frames))
    return frames, seperate_count


def is_similar(frame1, frame2, threshold=0.9):
    # Your similarity checking code here...
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hist_frame1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
    hist_frame2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

    hist_frame1 /= hist_frame1.sum()
    hist_frame2 /= hist_frame2.sum()

    intersection = cv2.compareHist(hist_frame1, hist_frame2, cv2.HISTCMP_INTERSECT)
    return intersection >= threshold


def frames_to_pdf(frames, output_pdf):
    # Your frames to PDF conversion code here...
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


def frames_to_pdf_stitching(frames, output_pdf):
    # Your frames to PDF conversion code here...
    stitcher1 = Stitcher(detector="sift", confidence_threshold=0.2)

    start = 0
    res_count = 0
    res_frames = []
    while start + 4 < len(frames):
        try:
            stitched = stitcher1.stitch(frames[start:start + 4])
            res_frames.append(stitched)
            res_count += 1
            start += 4
        except:
            start += 4

    # Add last frame
    res_frames.append(frames[-1])
    
    stitcher2 = Stitcher(detector="sift")

    final_res_frames = []
    start = 0
    break_condidtion = False
    page_break = 6
    while not break_condidtion:
        if start + page_break < len(res_frames):
            try:
                final_res_frames.append(res_frames[start:start + page_break])
                start += page_break
            except:
                pass
        else:
            final_res_frames.append(res_frames[start:])
            break_condidtion = True
    res_images = []
    for imgs in final_res_frames:
        if len(imgs) >= 3:
            try:
                res_images.append(stitcher2.stitch(imgs))
            except:
                for im in imgs:
                    res_images.append(im)
        else:
            for im in imgs:
                res_images.append(im)

    canvas_init_width, canvas_init_height = res_images[0].shape[0], res_images[0].shape[1]
    c = canvas.Canvas(output_pdf, pagesize=(canvas_init_width, canvas_init_height))
    print('Frames to PDF Started...')
    for idx, frame in enumerate(res_images):
        img_buffer = BytesIO()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
        frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
        frame_pil.save(img_buffer, format='JPEG')

        img_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        img_reader = ImageReader(img_buffer)
        c.drawImage(img_reader, 0, int(canvas_init_height - frame.shape[1]), width=frame.shape[0], height=frame.shape[1])
        c.showPage()

    c.save()
    print('PDF with frames created successfully!')


if __name__ == '__main__':
    VideoPlayerApp().run()
