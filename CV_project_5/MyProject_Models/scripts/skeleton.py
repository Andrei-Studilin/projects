import matplotlib.pyplot as plt
import cv2
import numpy as np

def draw_skeleton_per_person(frame_name, img, all_keypoints, all_scores, confs, keypoint_threshold=2,
                             conf_threshold=0.9):
    img_name = frame_name
    
    keypoints = ['nose','left_eye','right_eye',
             'left_ear','right_ear','left_shoulder',
             'right_shoulder','left_elbow','right_elbow',
             'left_wrist','right_wrist','left_hip',
             'right_hip','left_knee', 'right_knee',
             'left_ankle','right_ankle']
    
    limbs = [
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
        [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
        [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
    ]
    
    img_copy = img.copy()
    if len(all_keypoints)>0:
      colors = (0,0,255)
      for person_id in range(len(all_keypoints)):
          if confs[person_id]>conf_threshold:
            keypoints = all_keypoints[person_id, ...]

    
            for limb_id in range(len(limbs)):
              # выберите первоначальную точку конечности 1
              limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().cpu().numpy().astype(np.int32)
              # выберите первоначальную точку конечности 2
              limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().cpu().numpy().astype(np.int32)
              # рассматривайте оценку уверенности конечностей как минимальную оценку ключевой точки
              # среди двух оценок ключевых точек
              limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
              # проверяем, если скор конечностей превосходит порог
              if limb_score> keypoint_threshold: 
                # рисуем линию шириной 3
                cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), colors, 3)

    return cv2.imwrite(img_name, img_copy)