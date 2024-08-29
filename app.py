from flask import Flask, render_template, request, jsonify, Response
import firebase_admin
from firebase_admin import credentials, storage
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

app = Flask(__name__)

# Initialize Firebase Admin
cred = credentials.Certificate(r"E:\flask project\ar-writing-gesture-firebase-adminsdk-x8jl3-08facc9920.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ar-writing-gesture.appspot.com'
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    bucket = storage.bucket()
    blob = bucket.blob(f'drawings/{filename}')

    # Ensure unique filename by checking if it already exists
    if blob.exists():
        base, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        filename = f"{base}_{timestamp}{ext}"
        blob = bucket.blob(f'drawings/{filename}')

    blob.upload_from_file(file)
    return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200

@app.route('/control', methods=['POST'])
def control():
    action = request.json.get('action')
    if action in ['clear', 'save', 'pause', 'play']:
        return jsonify({'status': f'{action.capitalize()} action triggered'}), 200
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    points = [deque(maxlen=512)]
    index = 0
    color = (0, 0, 0)  # Black color for ink on the feed
    save_triggered = False
    save_paused = False
    file_count = 0

    # Ensure the static directory exists
    save_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = frame.copy()

        # Draw the buttons on the frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (200, 1), (300, 65), (0, 0, 0), 2)
        cv2.putText(frame, "SAVE", (215, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (360, 1), (460, 65), (0, 0, 0), 2)
        cv2.putText(frame, "PLAY", (375, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (520, 1), (620, 65), (0, 0, 0), 2)
        cv2.putText(frame, "PAUSE", (535, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                height, width, _ = frame.shape
                for lm in handslms.landmark:
                    lmx = int(lm.x * width)
                    lmy = int(lm.y * height)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            if (thumb[1] - center[1] < 30):
                points.append(deque(maxlen=512))
                index += 1
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    points = [deque(maxlen=512)]
                    index = 0
                    canvas[:, :] = frame[:, :]  # Clear canvas to the current frame
                elif 200 <= center[0] <= 300:  # Save Button
                    if not save_paused:
                        save_triggered = True
                elif 360 <= center[0] <= 460:  # Play Button
                    save_paused = False  # Resume saving
                elif 520 <= center[0] <= 620:  # Pause Button
                    save_paused = True  # Pause saving
            else:
                if 0 <= index < len(points):
                    points[index].appendleft(center)
        else:
            points.append(deque(maxlen=512))
            index += 1

        # Draw lines on the canvas
        for i in range(len(points)):
            for j in range(1, len(points[i])):
                if points[i][j - 1] is None or points[i][j] is None:
                    continue
                cv2.line(canvas, points[i][j - 1], points[i][j], color, 2)

        # Draw the overlay buttons on the canvas
        canvas = cv2.rectangle(canvas, (40, 1), (140, 65), (0, 0, 0), 2)
        canvas = cv2.putText(canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = cv2.rectangle(canvas, (200, 1), (300, 65), (0, 0, 0), 2)
        canvas = cv2.putText(canvas, "SAVE", (215, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = cv2.rectangle(canvas, (360, 1), (460, 65), (0, 0, 0), 2)
        canvas = cv2.putText(canvas, "PLAY", (375, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = cv2.rectangle(canvas, (520, 1), (620, 65), (0, 0, 0), 2)
        canvas = cv2.putText(canvas, "PAUSE", (535, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', canvas)
        frame = buffer.tobytes()

        if save_triggered:
            file_count = 1
            filename = f"painting_{file_count}.jpg"
            filepath = os.path.join(save_dir, filename)

            # Ensure unique filename
            while os.path.exists(filepath):
                file_count += 1
                filename = f"painting_{file_count}.jpg"
                filepath = os.path.join(save_dir, filename)

            print(f"Saving image to {filepath}")
            # Create a black canvas for saving
            black_canvas = np.zeros_like(canvas)

            # Draw white lines on the black canvas
            for i in range(len(points)):
                for j in range(1, len(points[i])):
                    if points[i][j - 1] is None or points[i][j] is None:
                        continue
                    cv2.line(black_canvas, points[i][j - 1], points[i][j], (255, 255, 255), 2)

            # Save the image with a black background and white ink
            cv2.imwrite(filepath, black_canvas)
            bucket = storage.bucket()
            blob = bucket.blob(f'drawings/{filename}')  # Upload to 'drawings' directory
            blob.upload_from_filename(filepath)
            save_triggered = False
            save_paused = True  # Pause saving after saving an image

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
