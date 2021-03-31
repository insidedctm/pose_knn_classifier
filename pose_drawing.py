import cv2

def draw_keypoints(frame, keypoints, join_keypoints=False):
  if join_keypoints:
    for start, end in zip(keypoints[:-1], keypoints[1:]):
      draw_line(frame, start, end)

  for keypoint in keypoints:
    draw_keypoint(frame, keypoint)
  return frame

def draw_keypoint(frame, keypoint):
  cv2.circle(frame, tuple(keypoint), 3, (0,0,255), -1)

def draw_line(frame, start, end):
  cv2.line(frame,tuple(start),tuple(end),(255,0,0),2)  

if __name__ == '__main__':
  video = '/Users/robineast/Dropbox/WithU/plank to t - hips in line with body.mp4'
  cap = cv2.VideoCapture(video)
  ret, frame = cap.read()
  if ret:
    draw_keypoints(frame, [[32, 140], [59,221]], False)
    cv2.imshow('pose drawing', frame)
    cv2.waitKey(0)
