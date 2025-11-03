from flask import Flask, render_template, Response, send_from_directory, request
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import time
from collections import deque
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Create directories if they don't exist
if not os.path.exists('static/recordings'):
    os.makedirs('static/recordings')
if not os.path.exists('static/thumbnails'):
    os.makedirs('static/thumbnails')

# In-memory state
cameras = {}

def load_events():
    if os.path.exists('events.json'):
        with open('events.json', 'r') as f:
            return json.load(f)
    return []

def save_events(events):
    with open('events.json', 'w') as f:
        json.dump(events, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/notifications')
def notifications_page():
    events = load_events()
    return render_template('notifications.html', events=events)

@app.route('/view/<event_id>')
def view_event_page(event_id):
    events = load_events()
    event = next((e for e in events if e['id'] == event_id), None)
    if not event:
        return "Event not found", 404
    return render_template('view.html', event=event)

@socketio.on('frame')
def handle_frame(message):
    camera_name = message['name']
    sid = request.sid
    data_url = message['data']

    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)

    np_arr = np.frombuffer(binary_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # Live stream to viewers
    _, buffer = cv2.imencode('.jpg', frame)
    socketio.emit('live_feed', {'name': camera_name, 'data': base64.b64encode(buffer).decode('utf-8')})

    if camera_name not in cameras:
        cameras[camera_name] = {
            'sid': sid,
            'last_frame': None,
            'motion_detected': False,
            'video_writer': None,
            'highlight_writer': None,
            'event_id': None,
            'last_motion_time': 0,
            'frame_buffer': deque(maxlen=600), # 60 seconds buffer at 10 FPS
            'lock': threading.Lock()
        }
    elif cameras[camera_name]['sid'] != sid:
        cameras[camera_name]['sid'] = sid

    cam = cameras[camera_name]
    with cam['lock']:
        # Add frame to buffer
        cam['frame_buffer'].append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if cam['last_frame'] is None:
            cam['last_frame'] = gray
            return

        frame_delta = cv2.absdiff(cam['last_frame'], gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False
        for contour in contours:
            if cv2.contourArea(contour) < 500: # Min contour size
                continue
            motion_found = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_found:
            cam['last_motion_time'] = time.time()
            if not cam['motion_detected']:
                print(f"Motion detected on {camera_name}")
                cam['motion_detected'] = True
                event_id = f"{camera_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cam['event_id'] = event_id

                video_path = f"static/recordings/{event_id}.avi"
                highlight_path = f"static/recordings/{event_id}_highlight.avi"

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                cam['video_writer'] = cv2.VideoWriter(video_path, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
                cam['highlight_writer'] = cv2.VideoWriter(highlight_path, fourcc, 10.0, (frame.shape[1], frame.shape[0]))

                # Write the buffer to the video files
                for f in list(cam['frame_buffer']):
                    cam['video_writer'].write(f)
                    # For the highlight video, we need to re-process the frame to add rectangles
                    gray_buffered = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    gray_buffered = cv2.GaussianBlur(gray_buffered, (21, 21), 0)
                    frame_delta_buffered = cv2.absdiff(cam['last_frame'], gray_buffered)
                    thresh_buffered = cv2.threshold(frame_delta_buffered, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh_buffered = cv2.dilate(thresh_buffered, None, iterations=2)
                    contours_buffered, _ = cv2.findContours(thresh_buffered.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours_buffered:
                        if cv2.contourArea(contour) < 500:
                            continue
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cam['highlight_writer'].write(f)

                events = load_events()
                new_event = {
                    'id': event_id,
                    'camera_name': camera_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'recording'
                }
                events.insert(0, new_event)
                save_events(events)
                socketio.emit('new_event', new_event)

        if cam['motion_detected']:
            cam['video_writer'].write(cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
            cam['highlight_writer'].write(frame)

            if time.time() - cam['last_motion_time'] > 5:
                print(f"Stopping recording for {camera_name}")
                cam['motion_detected'] = False
                cam['video_writer'].release()
                cam['highlight_writer'].release()

                events = load_events()
                for e in events:
                    if e['id'] == cam['event_id']:
                        e['status'] = 'finished'
                        break
                save_events(events)
                socketio.emit('event_finished', {'id': cam['event_id']})

        cam['last_frame'] = gray

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    disconnected_camera_name = None
    for name, cam_data in cameras.items():
        if cam_data.get('sid') == sid:
            disconnected_camera_name = name
            break

    if disconnected_camera_name:
        cam = cameras[disconnected_camera_name]
        if cam.get('motion_detected'):
            print(f"Camera {disconnected_camera_name} disconnected during recording. Finalizing.")
            cam['motion_detected'] = False
            cam['video_writer'].release()
            cam['highlight_writer'].release()

            events = load_events()
            for e in events:
                if e['id'] == cam['event_id']:
                    e['status'] = 'finished'
                    break
            save_events(events)
            socketio.emit('event_finished', {'id': cam['event_id']})

    print(f"Client disconnected: {sid}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
