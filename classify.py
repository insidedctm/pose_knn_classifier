import argparse
import pandas as pd
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
from pose_embedder import FullBodyPoseEmbedder
from pose_classifier import PoseClassifier
import numpy as np

classifiers = {}

def run_classify(csv_path):
  # initialise Pose estimator for whole video
  pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
  )

  print(f'Reading from {csv_path}')

  df = pd.read_csv(csv_path)
 
  filepaths_exercises = zip(df['filepaths'], df['exercise'], df['groundtruth']) 
  classifications = [classify(fname, exercise, gt, pose)  for fname, exercise, gt in filepaths_exercises]

  df['prediction'] = classifications

  df.to_csv(csv_path, header=True, index=None)

def classify(fname, exercise, groundtruth, pose):
  classifier_samples_folder = f'{exercise}_csvs_out'

  # Transforms pose landmarks into embedding.
  pose_embedder = FullBodyPoseEmbedder()

  # Classifies give pose against database of poses.

  if classifier_samples_folder in classifiers:
    pose_classifier = classifiers[classifier_samples_folder]
  else:
    pose_classifier = PoseClassifier(
      pose_samples_folder=classifier_samples_folder,
      pose_embedder=pose_embedder,
      top_n_by_max_distance=30,
      top_n_by_mean_distance=10)
    classifiers[classifier_samples_folder] = pose_classifier

  print(fname)
  print(exercise)
  img = cv2.imread(fname)

  classification_result = 0.0  
  results = pose.process(img)
  pose_landmarks = results.pose_landmarks  
  if pose_landmarks:

    frame_height, frame_width = img.shape[0], img.shape[1]
    pose_landmarks = np.array(
            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
               for lmk in pose_landmarks.landmark],
              dtype=np.float32)
    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
    p_w_bar_x = {k:v/10. for k,v in sorted(pose_classifier(pose_landmarks).items(), key=lambda item: item[1], reverse=True)}
    print(f'P(w|x): {p_w_bar_x}')
    print(groundtruth)
    gt_label = f'{exercise}_{groundtruth}'
    if gt_label in p_w_bar_x:
      classification_result = float(p_w_bar_x[gt_label])

  return classification_result

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('csv_path')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(args)

  run_classify(args.csv_path)

