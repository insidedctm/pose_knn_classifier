import argparse
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
from rep_counter import RepetitionCounter
from constants import Keypoints
import numpy as np
import airtable

class RepCounter:
  def __init__(self):
    self.n_reps = 0

  def __call__(self, pose):
    pass


class CorrectionsManager:
  def __init__(self):
    self.n_reps = 0
    self.correction_text = 'Hips in line with your body'
    self.colour          = (0, 255, 0)

  def __call__(self, pose):
    pass

  def set_reps(self, reps):
    self.n_reps = reps


def demo_corrections(video_path, limbs, metric_type, when, trigger):
  print(f'Demo Corrections for {video_path}')

  # setup rep counter and corrections managers
  counter = RepetitionCounter(class_name='plank_to_t_up')
  corrections = [CorrectionsManager()]

  # initialise Pose estimator for whole video
  pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
  )

  video = cv2.VideoCapture(video_path)
  while True:
    ret, frame = video.read()
    if not ret:
      break
    (h,w) = frame.shape[0], frame.shape[1]
    results = pose.process(frame)
    pose_landmarks = results.pose_landmarks
    if pose_landmarks is None:
      continue

#    # pass pose into rep counter
#    counter(pose)
#    
#    # pass pose into each corrections tracker
#    for tracker in corrections:
#      # first update the tracker for which rep we are on
#      tracker.set_reps(counter.n_reps)
#      # pass pose into tracker
#      tracker(pose)
#
#    frame = overlay_reps(frame, counter.n_reps)
#    frame = overlay_corrections(
#                      frame, 
#                      [tracker.correction_text  for tracker in corrections], 
#                      [tracker.colour for tracker in corrections]
#    )

    keypoints = filter_keypoints(pose_landmarks, limbs)
    metric = calculate_metric(keypoints, metric_type, (w,h))
    frame = overlay_metric(frame, limbs, metric_type, metric)

    cv2.imshow('corrections', frame)
    cv2.waitKey(25)

KEYPOINTS = {
  'LEFT_HIP_JOINT': [int(Keypoints.LEFT_SHOULDER), int(Keypoints.LEFT_HIP), int(Keypoints.LEFT_KNEE)],
  'LEFT_SHOULDER': [int(Keypoints.LEFT_SHOULDER)],
  'LEFT_WRIST': [int(Keypoints.LEFT_WRIST)]
}

def filter_keypoints(landmarks, limbs):
  filter = [kp for l in limbs for kp in KEYPOINTS[l]] 
  lmk = np.array(landmarks.landmark)
  return lmk[filter]

def calculate_metric(keypoints, metric_type, shape, use_3d=False):
  metric = 0.
  if metric_type == 'ANGLE':
    metric = calculate_angle_metric(keypoints, use_3d)
  elif metric_type == 'DISTANCE':
    metric = calculate_distance_metric(keypoints, shape, use_3d)
  elif metric_type == 'DISTANCE_X':
    metric = calculate_distance_x_metric(keypoints, shape)
  else:
    print('unknown metric')
  return metric

def calculate_distance_x_metric(keypoints, shape):
  w, h = shape
  if not len(keypoints) == 2:
    print(f'should be 2 keypoints, found {len(keypoints)}')
    return 0.
  end1 = keypoints[0]
  end2 = keypoints[1]
  dist = np.abs(end1.x*w - end2.x*w)

  return dist


def calculate_angle_metric(keypoints, use_3d=False):
  end1 = keypoints[0]
  end2 = keypoints[2]
  common = keypoints[1]
  v1 = np.array([end1.x - common.x, end1.y - common.y])
  if use_3d:
    v1 = np.array([end1.x - common.x, end1.y - common.y, end1.z - common.z])
  v2 = np.array([end2.x - common.x, end2.y - common.y])
  if use_3d:
    v2 = np.array([end2.x - common.x, end2.y - common.y, end2.z - common.z]) 
  angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
  angle = angle / np.pi * 180

  return angle

def calculate_distance_metric(keypoints, shape, use_3d=False):
  w, h = shape
  if not len(keypoints) == 2:
    print(f'should be 2 keypoints, found {len(keypoints)}')
    return 0.
  end1 = keypoints[0]
  end2 = keypoints[1]
  v1 = np.array([(end1.x - end2.x)*w, (end1.y - end2.y)*h])
  if use_3d:
    v1 = np.array([(end1.x - end2.x)*w, (end1.y - end2.y)*h, (end1.z - end2.z)*w])
  dist = np.linalg.norm(v1)

  return dist

def overlay_metric(frame, limbs, metric_type, metric):
  print(metric)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.75
  colour = (0, 255, 0)
  thickness = 2
  line_type = cv2.LINE_AA

  text = f'{limbs} ({metric_type})'
  pos = (10, 30) 
  frame = write_text(frame, text, pos, font, font_scale, colour, thickness, line_type)

  text = str(round(metric, 0))
  pos = (10, 60)
  frame = write_text(frame, text, pos, font, font_scale, colour, thickness, line_type)

  return frame

def write_text(frame, text, pos, font, font_scale, colour, thickness, line_type):
  cv2.putText(
		frame, 
		text, 
		pos, 
		font, 
		font_scale, 
		colour, 
		thickness, 
		line_type
  )
  return frame

def get_pose(frame):
  return None

def overlay_reps(frame, n_rep):
  return frame

def overlay_corrections(frame, texts, colours):
  return frame

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('video')
  parser.add_argument('--limbs', nargs='+', default='LEFT_HIP_JOINT')
  parser.add_argument('--metric', default='ANGLE')
  parser.add_argument('--when', default='ACROSS_EACH_REP')
  parser.add_argument('--trigger', default='MAX_LT')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  video = args.video
  if video.isnumeric():
    print(dir(airtable))
    video = airtable.download_video_by_ref(video)

  demo_corrections(video, args.limbs, args.metric, args.when, args.trigger)
