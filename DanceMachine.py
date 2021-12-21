import cv2
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from json import loads
import time
import multiprocessing
import playsound
from numpy import true_divide

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_drawing_style_dot_standard = mp_drawing.DrawingSpec(color=(80,80,80), thickness=2, circle_radius=2)
mp_drawing_style_line_standard = mp_drawing.DrawingSpec(color=(80,80,80), thickness=2, circle_radius=2)
mp_drawing_style_dot_Wrong = mp_drawing.DrawingSpec(color=(100,100,100), thickness=2, circle_radius=2)
mp_drawing_style_line_Wrong = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_drawing_style_dot_Right = mp_drawing.DrawingSpec(color=(100,100,0), thickness=2, circle_radius=2)
mp_drawing_style_line_Right = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
font = cv2.FONT_HERSHEY_SIMPLEX

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

index = [(11, 13),(13, 15), (12, 14), (14, 16), (24, 26), (26, 28), (23, 25), (25, 27)]
dir = "E:\\Development\\Python\\DanceMachine\\resource"

f = open(f'{dir}\\pose.json')
posedata = loads(f.read())
f.close()

posesec = list(posedata)
poseidx = list(posedata[posesec[0]])
posedx = list(posedata[posesec[0]])

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:


  frame = 0
  count = 0
  mode = [0.2, 0.09, 0.03] ## EASY, NORMAL, HARD
  print("EASY:0, NORMAL:1, HARD:2\nInput index number")
  
  while 1:
    num = int(input())
    if (num == 0):
      print("Easy mode seleced")
      break
    elif (num == 1):
      print("Normal mode seleced")
      break
    elif (num == 2):
      print("Hard mode seleced")
      break
    else:
      pass
  errp = mode[num]

  # webcam input:
  cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
  # video input
  bg = cv2.imread(f"{dir}\\bg1.jpg")
  vidcap = cv2.VideoCapture(f'{dir}\\sample.mp4')
  

  width, height = int(cap.get(3)), int(cap.get(4))
  bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)
  triggar = False

  def soundgo():
    os.system(f"{dir}\\sample.mp3")

  while vidcap.isOpened():
    if frame == 0:
      soundgo()
    mp_drawing_style_dot = mp_drawing_style_dot_standard
    mp_drawing_style_line = mp_drawing_style_line_standard

    success, img = cap.read()
    success2, src = vidcap.read()
    frame +=1
    nowsec = frame / 25
    if str(nowsec) in posesec:
      nowpose = posedata[str(nowsec)]
      recog = True
    else:
      recog = False

    standard = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
    
    image = segmentor.removeBG(img, bg, threshold=0.8)
    
    if not success:
      print("Ignoring empty camera frame.")
      continue
    if not success2:

      print("Ignoring empty video frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)


    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # when landmark exists.
    try: 
      # For extracting x,y coordinates
      keypoints = []
      for data_point in results.pose_landmarks.landmark: keypoints.append({'X': data_point.x, 'Y': data_point.y, 'Z': data_point.z, 'Visibility': data_point.visibility,})
    except: # when landmark dosen't exist.
      pass
    if recog == True:
      for idx1, idx2 in index:
        try:
          # Standard inclination
          M = (nowpose[idx2]['Y'] - nowpose[idx1]['Y']) / (nowpose[idx2]['X'] - nowpose[idx1]['X'])
          # User inclination
          m = -((keypoints[idx2]['Y'] - keypoints[idx1]['Y']) / (keypoints[idx2]['X'] - keypoints[idx1]['X']))
          if (M - errp <= m <= M + errp):
            count += 1
            mp_drawing_style_dot = mp_drawing_style_dot_Right
            mp_drawing_style_line = mp_drawing_style_line_Right
          else:
            mp_drawing_style_dot = mp_drawing_style_dot_Wrong
            mp_drawing_style_line = mp_drawing_style_line_Wrong
        except:
          pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing_style_dot, mp_drawing_style_line)
    text = f'points : {count * 10}'

    cv2.putText(standard, text, (50, 100), font, 1, (0,255,0),2)
    imgStack = cvzone.stackImages([standard, cv2.flip(image, 1)], 2,1)
    
    # print(f'points : {count * 10}')
    cv2.imshow('Dance Machine', imgStack)
    if cv2.waitKey(5) & 0xFF == 27:
      break
  os.system('taskkill /f /fi "windowtitle eq Groove 음악"')
  time.sleep(1)
  cap.release()
  vidcap.release()
