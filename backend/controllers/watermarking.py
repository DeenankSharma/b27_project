import cv2
import numpy as py

def embed_watermark(frame,watermark):
  watermark = cv2.resize(watermark,(frame.shape[1],frame.shape[0]))
  if len(watermark.shape) == 3:
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
  _, watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)
  frame = frame & ~1  
  frame[:, :, 0] |= watermark  
  return frame


def process_video(input_video_path,output_video_path,watermark_image_path):
    cap = cv2.VideoCapture(input_video_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        watermarked_frame = embed_watermark(frame, watermark)
        out.write(watermarked_frame)
    cap.release()
    out.release()

def extract_watermark(watermarked_frame):
   watermark = watermarked_frame[:,:,0] & 1
   return watermark * 255

def extract_from_video(watermarked_video_path):
  cap = cv2.VideoCapture(watermarked_video_path)
  while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
      break
    watermark = extract_watermark(frame)
    cv2.imshow('Extracted Watermark', watermark)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    cap.release()
    cv2.destroyAllWindows()
    