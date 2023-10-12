import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model and labels
model_path = '/Users/apple/Desktop/pyapi/assets/model_unquant.tflite'
labels_path = '/Users/apple/Desktop/pyapi/assets/labels.txt'

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(frame_resized, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run the inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_label = labels[np.argmax(output_data)]

    # Display the results
    cv2.putText(frame, pred_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Currency Detection', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
