import numpy as np
import math
from spherical_binning import get_bins, embed_index_from_bin
from scipy.stats import norm

class ViewInvariantPoseEmbedder(object):
  """Converts 3D pose landmarks into view invariant embedding."""
  KEYPOINTS = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

  HOJ3D_KEYPOINTS = [
        'nose',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    

  def __call__(self, landmarks):
    """converts to view invariant embedding. See Xia, View Invariant Human Action Recognition Using Histograms of 3D Joints (2012)
       s4.1/4.2
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      The embedding produces an 84-D vector. 
    """

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Get embedding.
    embedding = self._get_pose_distance_embedding(landmarks)

    return embedding

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self.KEYPOINTS.index('left_hip')]
    right_hip = landmarks[self.KEYPOINTS.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_azimuth_direction(self, landmarks):
    """Calculates pose azimuth direction as the vector from origin to """
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_shoulder_centre(self, landmarks):
    """Calculates centre of shoulder."""
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    center = (left_shoulder + right_shoulder) * 0.5
    return center

  def _get_spherical_from_landmarks(self, landmarks):
    print('_get_spherical_from_landmarks')
    print(landmarks)
    sphericals = []
    hip_centre = self._get_pose_center(landmarks)
    x0 = hip_centre[0]
    y0 = hip_centre[1]
    z0 = hip_centre[2]
    xL = landmarks[self.KEYPOINTS.index('left_hip')][0]
    xR = landmarks[self.KEYPOINTS.index('right_hip')][0]
    zL = landmarks[self.KEYPOINTS.index('left_hip')][2]
    zR = landmarks[self.KEYPOINTS.index('right_hip')][2]
    mag = math.sqrt(math.pow(xL-xR, 2)+math.pow(zL-zR, 2))
    href = [(xL-xR)/mag, 0, (zL-zR)/mag]
    vref = [0, 1, 0]
    for kp in self.HOJ3D_KEYPOINTS:
      print(kp)
      x = landmarks[self.KEYPOINTS.index(kp)][0] - x0
      y = landmarks[self.KEYPOINTS.index(kp)][1] - y0
      z = landmarks[self.KEYPOINTS.index(kp)][2] - z0
      alpha = math.acos(dotProduct([x,0,z], href)/getMagnitude([x,0,z]))
      print(alpha)
      theta = math.acos(dotProduct([x,y,z], vref)/getMagnitude([x,y,z]))
      while ((alpha <= -math.pi) | (alpha >= math.pi)):
        alpha+=-math.copysign(1, alpha)*2*math.pi
      while ((theta <= -math.pi) & (theta >= math.pi)):
        theta+=-math.copysign(1, theta)*2*math.pi
      print(f'appending {alpha}, {theta} to sphericals')
      sphericals.append([alpha, theta])

    return sphericals

  def _get_pose_distance_embedding(self, landmarks):
    """Converts pose landmarks into  embedding.
   
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with 84-Di pose embedding
    """
    #origin = _get_pose_center(landmarks)
    #shoulder_center = _get_shoulder_centre(landmarks)
    #azimuth = _get_azimuth_direction(landmarks, origin)
    #zenith = _get_zenith_direction(landmarks, origin)

    embedding = [0.] * 84

    sphericals = self._get_spherical_from_landmarks(landmarks)
    print(f'sphericals: {sphericals}')

    bin_boundaries = [get_bins(alpha, theta)  for alpha, theta in sphericals]

    scale = 0.5 # variance for normal cdf
    for (alpha, theta), bb in zip(sphericals, bin_boundaries):
      print(f'alpha={alpha}, theta={theta}, bin boundaries element: \n{bb}')
      for abounds in bb[0]: # alpha boundaries
        print(f'alpha bounds to calculate cdf on: {abounds}') 
        upper_cdf = norm.cdf(abounds[1], alpha, scale)
        print(f'  cdf on upper: {upper_cdf}')
        lower_cdf = norm.cdf(abounds[0], alpha, scale)
        print(f'  cdf on lower: {lower_cdf}')
        a_vote = upper_cdf - lower_cdf
        print(f'  vote = {a_vote}')
        for tbounds in bb[1]: # theta boundaries 
          print(f'theta bounds to calculate cdf on: {tbounds}') 
          upper_cdf = norm.cdf(tbounds[1], theta, scale)
          print(f'  cdf on upper: {upper_cdf}')
          lower_cdf = norm.cdf(tbounds[0], theta, scale)
          print(f'  cdf on lower: {lower_cdf}')
          t_vote = upper_cdf - lower_cdf
          print(f'  vote = {t_vote}')
          total_vote = a_vote * t_vote
          print(f'total_vote: {total_vote}')

          embedding_ix = embed_index_from_bin(abounds, tbounds)
          embedding[embedding_ix] += total_vote

    return embedding

def getMagnitude(vec):
    return math.sqrt(vec[0]*vec[0]+ vec[1]*vec[1] + vec[2]*vec[2])

def dotProduct(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def scalarMult(s, vec):
    return [s*vec[0], s*vec[1], s*vec[2]]
