import argparse
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX 

def extract_frames(input_video, output_folder, exercise):
  up_folder_path = f'{output_folder}/{exercise}_up'
  down_folder_path = f'{output_folder}/{exercise}_down'

  check_and_create_dir(up_folder_path)
  check_and_create_dir(down_folder_path)

  video = cv2.VideoCapture(input_video)

  cnt=0
  while True:
    ret, frame = video.read()
    if not ret:
      print('no more frames')
      break

    cnt += 1

    frame = rotate_and_resize(frame, target_width=320)

    cv2.imshow('video', withText(frame, f'{cnt}'))
    key = cv2.waitKey(0)
    print(key)
    if key & 0xFF == ord('q'):
      break
    if key & 0xFF == ord('u'):
      output_path = f'{up_folder_path}/{exercise}{cnt}.jpg'
      print(f'save to {output_path}')
      #frame = rotate_and_resize(frame, target_width=320) 
      cv2.imwrite(output_path, frame)
    if key & 0xFF == ord('d'):
      output_path = f'{down_folder_path}/{exercise}{cnt}.jpg'
      print(f'save to {output_path}')
      #frame = rotate_and_resize(frame, target_width=320)
      cv2.imwrite(output_path, frame)

def check_and_create_dir(path):
  ''' Check if folder path exists and create folder'''
  from pathlib import Path
  Path(path).mkdir(parents=True, exist_ok=True)

def withText(img, msg):
  img_copy = img.copy()
  cv2.putText(img_copy,  
                msg,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
  return img_copy

def rotate_and_resize(img, target_width):
  height, width, _ = img.shape
  scaling = target_width / width
  target_height = int( height * scaling)
  dim = (target_width, target_height)
  print(f'resizing to {dim}')
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  #rotated = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
  #return rotated
  return resized

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_video')
  parser.add_argument('output_folder')
  parser.add_argument('exercise')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  extract_frames(args.input_video, args.output_folder, args.exercise)


