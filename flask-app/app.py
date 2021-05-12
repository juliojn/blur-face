from flask import Flask, render_template, Response
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import cv2
import os

folder = "../dnn"
model  = os.path.sep.join([folder, "deploy.prototxt"])
weight = os.path.sep.join([folder, "res10_300x300_ssd_iter_140000.caffemodel"])

net = cv2.dnn.readNet(model, weight)
video_input = cv2.VideoCapture(0)
frame = None
time.sleep(3.0)

app = Flask(__name__)

def blur_face(image, factor=2):
	(h, w) = image.shape[:2]
	k_w = int(w / factor)
	k_h = int(h / factor)

	if k_w % 2 == 0:
		k_w -= 1
	if k_h % 2 == 0:
		k_h -= 1

	return cv2.GaussianBlur(image, (k_w, k_h), 0)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
	return render_template("index.html")

@app.route('/blur_face', methods=['GET', 'POST'])
def blur_face():
	return Response(generate_frame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate_frame():
	while True:
		ret, frame = video_input.read()
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

		(flag, encodedImage) = cv2.imencode(".jpg", frame)
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
	app.run(debug = True, host = '0.0.0.0')
