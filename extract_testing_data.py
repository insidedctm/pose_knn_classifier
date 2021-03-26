import argparse
import boto3
import mediapipe as mp
mp_pose = mp.solutions.pose
import cv2
import numpy as np
import csv

class PoseGenerator:
  def __init__(self, path):
    self.results = []
    self.height  = 0
    self.width   = 0

    video = cv2.VideoCapture(path)
    with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

      while True:
        ret, image = video.read()
        if not ret:
          break
        self.height, self.width = image.shape[0], image.shape[1]
        result = pose.process(image)
        self.results.append(result)
  
  def get_poses(self):
    return self.results

  def get_shape(self):
    return self.height, self.width

def extract_testing_data(s3_bucket, s3_prefix):
  print(f'extracting from {s3_bucket}/{s3_prefix}')
  keys = get_keys_in_folder(s3_bucket, s3_prefix)

  landmarks = {}

  for ix, k in enumerate(keys):
    if k.split('/')[-1] == '':
      continue
    print(f'procesing {k}')
    path = download_file_from_s3(s3_bucket, k)
    pose_generator = PoseGenerator(path)
    poses = pose_generator.get_poses()
    height, width = pose_generator.get_shape()
    landmarks[k] = get_landmarks(poses, height, width)
    
  save_landmarks(landmarks)

def get_keys_in_folder(s3_bucket, s3_prefix):
  keys = []
  s3 = boto3.resource('s3')
  bucket = s3.Bucket(s3_bucket)

  for o in bucket.objects.filter(Delimiter=f'/{s3_prefix}'):
    keys.append(o.key)
  
  return keys

def download_file_from_s3(bucket, key):
  filename = f'downloaded_files/{key.split("/")[-1]}'
  print(f'downloading file to {filename}')

  s3 = boto3.client('s3')
  s3.download_file(bucket, key, filename)
 
  return filename

def get_landmarks(poses, frame_height, frame_width):
  video_pose_landmarks = [np.array(
                [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose.pose_landmarks.landmark],
                dtype=np.float32) for pose in poses if pose.pose_landmarks]
  shapes = list(set([landmarks.shape for landmarks in video_pose_landmarks]))
  assert len(shapes) == 1, 'landmarks contains different shapes, expecting 1'
  assert shapes[0] == (33, 3), 'Unexpected landmarks shape: {}'.format(shapes[0])
  video_pose_landmarks = [pose_landmarks.flatten().astype(np.str).tolist() for pose_landmarks in video_pose_landmarks]
  return video_pose_landmarks

def save_landmarks(landmarks):
  all_rows = []
  for k in landmarks.keys():
    exercise_correction = k.split('/')[-1]
    exercise_correction = exercise_correction[:-4]
    rows = [[exercise_correction]+row for row in landmarks[k]]
    all_rows.extend(rows)
  with open('output.csv', 'w') as csv_out_file:
    for row in all_rows:
      csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      csv_out_writer.writerow(row)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('s3_bucket')
  parser.add_argument('s3_prefix')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  extract_testing_data(args.s3_bucket, args.s3_prefix)
