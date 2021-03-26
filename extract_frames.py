import argparse
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX 

def extract_frames(input_video, output_folder, exercise, keymap):
  video_name = input_video.split('/')[-1]

  keys, suffixes = parse_keymap(keymap)
  folder_map = {key:f'{output_folder}/{exercise}_{suffix}' for key, suffix in zip(keys,suffixes)}

  for folder_path in folder_map.values():
    check_and_create_dir(folder_path)

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
    if key & 0xFF == ord('q'):
      break
    if key & 0xFF in [k for k in keys]:
      folder_path = folder_map[key & 0xFF]
      output_path = f'{folder_path}/{video_name}_{exercise}{cnt}.jpg'
      print(f'save to {output_path}')
      cv2.imwrite(output_path, frame)

def parse_keymap(keymap):
  ''' Receives a string in <key>=<suffix>;<key>=<suffix>;... format
      and returns a list of keys and suffixes. The key is a single letter
      typed on the keyboard, the list of keys should be the ordinal value
      of the key
  '''
  entries = keymap.split(';')
  keys = [ord(entry.split('=')[0]) for entry in entries]
  suffixes = [entry.split('=')[1] for entry in entries]
  return keys, suffixes
  
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
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  #rotated = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
  #return rotated
  return resized

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_video')
  parser.add_argument('output_folder')
  parser.add_argument('exercise')
  parser.add_argument('--keymap', default='u=up;d=down', 
      help='map keys to class suffxies in key=suffix format (suffix is alphanumeric only')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(f'Using keymap: {args.keymap}')

  extract_frames(args.input_video, args.output_folder, args.exercise, args.keymap)


