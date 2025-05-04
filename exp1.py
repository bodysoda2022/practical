
#serial processing

import cv2

def process_frame_serial(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def process_video_serial(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame_serial(frame)
        if out is None:
            height, width = processed_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height), isColor=False)
        out.write(processed_frame)
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

process_video_serial("SampleVideo.mp4", "SerialProcessing.avi")

# parallel processing

from concurrent.futures import ThreadPoolExecutor
import cv2

def process_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def parallel_processing(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    with ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(process_frame, frames))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                          (processed_frames[0].shape[1], processed_frames[0].shape[0]), 
                          False)
    for frame in processed_frames:
        out.write(frame)
    out.release()

parallel_processing("SampleVideo.mp4", "ParallelProcessing.avi")

# pipeline processing

import cv2
import threading
import queue

def capture_frames(video_path, frame_queue):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)
    cap.release()

def process_frames(frame_queue, processed_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            processed_queue.put(None)
            break
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_queue.put(processed_frame)

def save_processed_frames(processed_queue, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    while True:
        processed_frame = processed_queue.get()
        if processed_frame is None:
            break
        if out is None:
            height, width = processed_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height), isColor=False)
        out.write(processed_frame)
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

def process_video_pipeline(video_path, output_path):
    frame_queue = queue.Queue()
    processed_queue = queue.Queue()
    t1 = threading.Thread(target=capture_frames, args=(video_path, frame_queue))
    t2 = threading.Thread(target=process_frames, args=(frame_queue, processed_queue))
    t3 = threading.Thread(target=save_processed_frames, args=(processed_queue, output_path))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

process_video_pipeline("SampleVideo.mp4", "PipelineProcessing.avi")
