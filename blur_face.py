from imutils.video import VideoStream
import numpy as np
import imutils
import time
import time
import cv2
import os

def blur_face(image, factor=2):
	(h, w) = image.shape[:2]
	k_w = int(w / factor)
	k_h = int(h / factor)

	if k_w % 2 == 0:
		k_w -= 1
	if k_h % 2 == 0:
		k_h -= 1

	return cv2.GaussianBlur(image, (k_w, k_h), 0)

# camera = VideoStream(0)
camera = cv2.VideoCapture("video.webm")
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))

folder = "dnn"
prototxtPath = os.path.sep.join([folder, "deploy.prototxt"])
weightsPath = os.path.sep.join([folder, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

start = time.time()
ret, frame = camera.read()

while ret:
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			face = blur_face(face)
			frame[startY:endY, startX:endX] = face
	output.write(frame)
	ret, frame = camera.read()

end = time.time()

print("\n", (end - start), " seconds\n")
