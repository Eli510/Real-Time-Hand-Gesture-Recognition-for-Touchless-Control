# Script to run inferences on Raspberry Pi while sending results to a redis server for remote viewing

import cv2
import tensorflow as tf
from picamera2 import Picamera2
import numpy as np
import multiprocessing as mp
import time
import ImageTransferService
# Code for sending images through redis obtained from: https://stackoverflow.com/questions/57876639/cvimshow-over-ssh-x11-too-slow-alternatives

model_path = 'model_3d.h5'
inference_time_csv = 'inference_time.csv'
gesture_to_index = {
    "Swiping Left": 0,
    "Swiping Right": 1,
    "Swiping Down": 2,
    "Swiping Up": 3,
    "Stop Sign": 4,
    "Thumb Down": 5,
    "Thumb Up": 6,
    "Doing other things": 7,
    "Drumming Fingers": 8,
    "No gesture": 9,
    "Pulling Hand In": 10,
    "Pulling Two Fingers In": 11,
    "Pushing Hand Away": 12,
    "Pushing Two Fingers Away": 13,
    "Rolling Hand Backward": 14,
    "Rolling Hand Forward": 15,
    "Shaking Hand": 16,
    "Sliding Two Fingers Down": 17,
    "Sliding Two Fingers Left": 18,
    "Sliding Two Fingers Right": 19,
    "Sliding Two Fingers Up": 20,
    "Turning Hand Clockwise": 21,
    "Turning Hand Counterclockwise": 22,
    "Zooming In With Full Hand": 23,
    "Zooming In With Two Fingers": 24,
    "Zooming Out With Full Hand": 25,
    "Zooming Out With Two Fingers": 26
}
index_to_gesture = {v: k for k, v in gesture_to_index.items()}


def main():
    # Load the model
    model = tf.keras.models.load_model(model_path)
    labels = index_to_gesture

    # Initialize shared variables
    video = mp.Manager().list()  # Shared list for video frames
    label = mp.Manager().Value(int, 0)  # Shared variable for label

    # Start the video capture process
    capture_process = mp.Process(target=capture_video, args=(video, labels, label))
    capture_process.start()
    print("Capture process started.")
    while True:
        if not capture_process.is_alive():
            print("Capture process terminated.")
            break

        # Check if there are enough frames for inference
        if len(video) >= 20:
            processing_time = inference(video, model, labels, label)
            # Save inference time to CSV
            with open(inference_time_csv, 'a') as f:
                f.write(f"{processing_time}\n")
            print(f"Inference time: {processing_time:.4f} seconds")


def capture_video(video, labels, label):
    # Initialize the remote display
    host = '192.168.137.59'
    RemoteDisplay = ImageTransferService.ImageTransferService(host)
    print(RemoteDisplay.ping())

    # Initialize the camera
    FPS = 30
    cap = Picamera2()
    config = cap.create_preview_configuration(main={"size": (640, 480)})
    cap.configure(config)
    cap.set_controls({"FrameRate": FPS})
    cap.start()
    print("Camera started.")

    video_length = 2  # Seconds of video to capture
    num_frames = int(video_length * FPS)

    while True:
        frame = cap.capture_array()
        if frame is None:
            print('Error: failed to capture frame')
            break

        # Convert the frame to opencv format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if len(video) >= num_frames:
            video.pop(0)
        # Preprocess the frame and add it to the video list
        video.append(preprocess_frame(frame, new_size=(64, 96)))

        # Add label to frame
        label_index = label.value
        label_text = labels[label_index]
        cv2.putText(frame, f"Gesture: {label_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send the frame to the remote display
        RemoteDisplay.sendImage(frame)


def inference(video, model, labels, label):
    num_frames = 20
    frames = np.array(video)
    sampled_frames = np.array([frames[i*len(frames) // num_frames] for i in range(0, num_frames)])
    sampled_frames = np.expand_dims(sampled_frames, axis=0)

    # Make predictions
    start_time = time.time()
    predictions = model.predict(sampled_frames, verbose=0)
    end_time = time.time()
    processing_time = end_time - start_time
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_class]

    # Update the shared label variable
    label.value = predicted_class

    # Display the prediction and confidence
    print("Predicted label:", predicted_label)
    print("Confidence:", predictions[0][predicted_class])
    return processing_time


def preprocess_frame(frame, new_size=(112, 112), mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    """
    1. Resizes the frame to new_size (width, height).
    2. Converts pixel values from [0..255] to [0..1].
    3. Normalizes each channel by mean and std (like ImageNet).
    4. Returns the processed frame as a NumPy array in shape (H, W, C).
    """
    # frame is a NumPy array: (H, W, 3) in BGR or RGB
    # 1. Resize
    resized = cv2.resize(frame, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)

    # 2. Convert to float32 and scale to [0..1]
    resized = resized.astype(np.float32) / 255.0

    # 3. Normalize: (pixel - mean) / std
    # mean and std are tuples for (R, G, B) or (B, G, R), ensure correct channel order
    # If the frame is in RGB, reorder the mean/std accordingly
    for c in range(3):
        resized[..., c] = (resized[..., c] - mean[c]) / std[c]

    return resized


if __name__ == '__main__':
    main()
