from pose_embedder import FullBodyPoseEmbedder
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
import numpy as np
from view_invariant_embedder import ViewInvariantPoseEmbedder
from spherical_binning import get_bins
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from time import time

train_example_path = '/Users/robineast/projects/pose_knn_classifier/fitness_poses_images_in/goblet_squat_up/goblet_squat99.jpg'

# initialise Pose estimator for whole video
pose = mp_pose.Pose(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5
)

embedder1 = FullBodyPoseEmbedder()
embedder2 = ViewInvariantPoseEmbedder()

def test_invariant_embedder():
  train_pose, img = get_pose_from_image(train_example_path)

  t0 = time()
  embedding1 = embedder1(train_pose)
  elapsed1 = time() - t0
  print(embedding1)

  sphericals = embedder2._get_spherical_from_landmarks(train_pose)
  print(f'sphericals: {sphericals}')

  print('')
  bin_boundaries = [get_bins(alpha, theta) for alpha, theta in sphericals]
  for bb in bin_boundaries:
    print(bb)

  print('''
  #
  # Test the embedding
  #
  ''')
  t0 = time()
  embedding2 = embedder2(train_pose)
  elapsed2 = time() - t0
  print(f'elapsed1 {elapsed1}sec')
  print(f'elapsed2 {elapsed2}sec')

  display_embedding(embedding2)

  fig_img = get_embedding_fig_img(embedding2)
  fig_img = cv2.resize(fig_img, (32,48))
  h, w = fig_img.shape[0:2]
  img[:h,:w] = fig_img
  
  cv2.imshow('embedding', img)
  cv2.waitKey(0)

def get_embedding_fig_img(data):
  fig = get_embedding_fig(data)
  plt.savefig('temp.png')
  img = cv2.imread('temp.png')
  return img

def fig2data( fig ):
  fig.canvas.draw()
  w,h = fig.canvas.get_width_height()
  #buf = np.fromstring( fig.canvas.tostring_argb(), dtype=np.uint8)
  buf = fig.canvas.tobuffer_argb()
  buf.shape = ( w, h, 4 )
  buf = np.roll( buf, 3, axis=2)
  return buf

def get_embedding_fig(data):
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.bar(range(84), data)
  return fig

def display_embedding(data):
  fig = get_embedding_fig(data)
  plt.show()

def get_pose_from_image(path):
  img = cv2.imread(path)
  t0 = time()
  results = pose.process(img)
  elapsed = time() - t0
  print(f'pose estimation elapsed {elapsed}sec')
  input('press any key ...')
  pose_landmarks = results.pose_landmarks
  if pose_landmarks:

    frame_height, frame_width = img.shape[0], img.shape[1]
    pose_landmarks = np.array(
            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
               for lmk in pose_landmarks.landmark],
              dtype=np.float32)
    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

  return pose_landmarks, img


if __name__ == '__main__':
  test_invariant_embedder()
