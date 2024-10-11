# opencv-python
# Pillow
# scipy
#from video_invisible_watermark import embed_watermark_to_video,extract_watermark_from_video
import cv2
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
from google.cloud import vision
from google.oauth2 import service_account 

# Replace 'video.mp4' with the path to your video file
# video_path = 'test_video.mkv'
# video = cv2.VideoCapture(video_path)

# # Get the frames per second (fps)
# fps = video.get(cv2.CAP_PROP_FPS)

# # Get the total number of frames
# total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

# # Calculate duration in seconds
# duration = total_frames // fps if fps > 0 else 0

# # Print the results

# # Release the video capture object
# video.release()

import numpy as np
from scipy.fftpack import dct, idct
import os


def load_image(image_path, size=(100, 100)):
    """加载并调整图像大小"""
    img = Image.open(image_path).convert('L')
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def embed_watermark_to_frame(frame, img):
    """将图像嵌入到整个帧（使用DCT）"""
    rows, cols, _ = frame.shape
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_frame)

    # 对 Y 组件进行 DCT 变换
    dct_y = dct(dct(y.T, norm='ortho').T, norm='ortho')

    # 将图像嵌入 DCT 系数中
    img_rows, img_cols = img.shape
    y_start, x_start = (rows - img_rows) // 2, (cols - img_cols) // 2
    dct_y[y_start:y_start + img_rows, x_start:x_start + img_cols] = img

    # 对 Y 组件进行逆 DCT 变换
    idct_y = idct(idct(dct_y.T, norm='ortho').T, norm='ortho')
    y = np.clip(idct_y, 0, 255).astype(np.uint8)  # 确保像素值在0-255之间

    # 合并 Y, U, V 组件
    yuv_frame = cv2.merge((y, u, v))
    frame_with_message = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
    return frame_with_message


def extract_watermark_from_frame(frame, img_size=(100, 100)):
    """从整个帧提取图像（使用DCT）"""
    rows, cols, _ = frame.shape
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_frame)

    # 对 Y 组件进行 DCT 变换
    dct_y = dct(dct(y.T, norm='ortho').T, norm='ortho')

    # 提取 DCT 系数中的图像
    img_rows, img_cols = img_size
    y_start, x_start = (rows - img_rows) // 2, (cols - img_cols) // 2
    extracted_img = dct_y[y_start:y_start + img_rows, x_start:x_start + img_cols]
    extracted_img = np.round(extracted_img).astype(np.uint8)

    return extracted_img


def video_to_frames(video_path, duration=10, fps=30):
    """将视频分解为帧"""
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(fps * duration)
    for _ in range(total_frames):
        success, image = vidcap.read()
        if not success:
            break
        frames.append(image)
    vidcap.release()
    return frames


def frames_to_video(frames, output_path, fps, size):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()


def save_images(images, prefix='result/pre_compression'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for i, img in enumerate(images):
        img_path = os.path.join(prefix, f'{i}.jpg')
        img.save(img_path)


def embed_watermark_to_video(input_video_path, output_video_path, watermark_path, duration=10, fps=30):
    watermark_img = load_image(watermark_path)
    frames = video_to_frames(input_video_path, duration)
    frames_with_message = [embed_watermark_to_frame(frame, watermark_img) for frame in frames]
    size = (frames[0].shape[1], frames[0].shape[0])
    frames_to_video(frames_with_message, output_video_path, fps, size)


def extract_watermark_from_video(output_video_path, prefix='compression', duration=10):
    output_frames = video_to_frames(output_video_path, duration)  # 解析前2秒
    post_compression_imgs = [extract_watermark_from_frame(frame) for frame in output_frames]
    post_compression_texts = [Image.fromarray(img) for img in post_compression_imgs]
    save_images(post_compression_texts, prefix)

def create_text_image(text, width, height, initial_font_size=32):
    img = Image.new('L', (width, height), color=255)  
    draw = ImageDraw.Draw(img)
    font_size = initial_font_size
    while True:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if text_width <= width and text_height <= height:
            break
        font_size -= 2
        if font_size <= 10: 
            break
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=0, font=font)

    return np.array(img)


def read_text_from_image(image_name, folder="compression"):
    """
    Sends an image from the 'compressions' folder to Google Cloud Vision (or a similar OCR service)
    and retrieves text from the image.

    Args:
    - image_name (str): The name of the image file (e.g., 'image.png').
    - folder (str): The folder containing the image (default is 'compressions').

    Returns:
    - str: The extracted text from the image.
    """
    # Set up the path to the image
    image_path = os.path.join(folder, image_name)
    
    credentials=service_account.Credentials.from_service_account_file("b27project-e4206a0ff48a.json")

    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient(credentials=credentials)
    
    # Load the image from file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # Create an image object for Vision API
    image = vision.Image(content=content)
    
    # Perform text detection (OCR)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Error with OCR API: {response.error.message}")
    
    # Return the detected text (first item usually contains the full text)
    if texts:
        return texts[0].description
    else:
        return "No text detected in the image."

# video = cv2.VideoCapture("test_video.mkv")
# fps = video.get(cv2.CAP_PROP_FPS)
# total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
# duration = total_frames // fps if fps > 0 else 0
# video.release()   
# img=create_text_image("67988998",200,100)
# cv2.imwrite("watermark_image.jpeg", img)

# embed_watermark_to_video("test_video.mp4","output_vid.mp4","watermark_image.jpeg",duration,fps)
# ffmpeg.input("output_vid.mp4").output("output_vid.mkv").run()
# ffmpeg.input("output_vid.mkv").output("output_vid_.mp4").run()
# extract_watermark_from_video("ouput_vid_.mp4",duration=5)

# image_name = "0.jpg"
# extracted_text = read_text_from_image(image_name)
# print("Extracted text:", extracted_text)
# # hZCWlyIAjJTOR6Et9WGFLs7gdFoWm+JHfZZfaqa0jmeOFcJQtoXlAHvPmo/yi9Ysufwq03uF0q3Lwm13sePRHljKWOsFCgLVZ0ZFlqDQOnTUlAxdWOCSdgqQuB5f+b1CHmE8Lz1seSG95QPHvMBcwG+9fZF4ac3M+Z+hTuXttgUAuJFbBcW0FxvasD+J8gxMmI5r3eEX5+uVR0tQDa5wqB89AWXYDygYRkLz/cbHwzmiIR+d3UU1lezzlnz3usU8zy/tB0+AomU/2tUSd5Uck0RFohJNvwhCUMrB4+RN/Tav0nguI5Pa3BZPX/6ciX1U1XafNPkD82Q2UDYZGR608g==
# #6 7 9 8 8 9 9 8