import argparse
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
from constants import Keypoints

FORWARD = 1
BACK    = 2

def view_video(path):
  # initialise Pose estimator for whole video
  pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
  )

  video = cv2.VideoCapture(path)
  frames = get_frames(video)
  cv2.namedWindow("view video")
  cv2.setMouseCallback("view video", capture_clicked_point)

  direction = FORWARD
  pos = 0
  while True:
    if pos > len(frames):
      pos = len(frames)
    if pos < 0:
      pos = 0
    frame = frames[pos] 
    (h,w) = frame.shape[0], frame.shape[1]
    cv2.imshow('view video', frame)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
      break
    if key & 0xFF == ord('f'):
      direction = FORWARD
    if key & 0xFF == ord('b'):
      direction = BACK

    if direction == FORWARD:
      pos += 1
    else:
      pos -= 1

def get_frames(video):
  frames = []
  while True:
    ret, frame = video.read()
    if not ret:
      break
    frames.append(frame)
  return frames

def capture_clicked_point(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    print(f'({x}, {y})')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('video_path')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  view_video(args.video_path)
