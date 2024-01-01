import cv2
import numpy as np

def filter_lines_by_angle(lines, min_angle, max_angle):
    if lines is None:
        return None
    new_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if min_angle <= angle <= max_angle or -max_angle <= angle <= -min_angle:
                new_lines.append([[x1, y1, x2, y2]])
    return new_lines

video = cv2.VideoCapture('/Users/calvinwak/Desktop/road.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture('/Users/calvinwak/Desktop/road.mp4')
        continue
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
  
    edges = cv2.Canny(blurred, 50, 150)
    
    
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, height), (0, height/2), (width, height/2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
    
  
    filtered_lines = filter_lines_by_angle(lines, 20, 160)
    

    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    

    cv2.imshow('frame', frame)
    
    
    key = cv2.waitKey(25)
    if key == 27:  
        break

video.release()
cv2.destroyAllWindows()

