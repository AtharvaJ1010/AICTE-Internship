import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import json

# Constants
DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
inWidth, inHeight = 368, 368

st.title("Advanced Human Pose Estimation")
st.markdown(
    """
    This app detects human poses and provides advanced tools for image enhancement, annotation, and visualization.  
    **Features**:
    - Pose keypoint visualization
    - Downloadable pose data (JSON format)
    """
)

threshold = st.slider("Key Point Detection Threshold", 0, 100, 20, 5) / 100
keypoint_color = st.color_picker("Key Point Color", "#FF0000")
line_color = st.color_picker("Line Color", "#00FF00")

st.subheader("Image Enhancement Tools")
brightness = st.slider("Adjust Brightness", 0.5, 3.0, 1.0)
contrast = st.slider("Adjust Contrast", 0.5, 3.0, 1.0)
sharpness = st.slider("Adjust Sharpness", 0.5, 3.0, 1.0)

st.subheader("Additional Features")
annotate_image = st.checkbox("Annotate with Pose Names", value=False)
convert_to_grayscale = st.checkbox("Convert Image to Grayscale", value=False)
rotation_angle = st.slider("Rotate Image (Degrees)", -180, 180, 0, 5)

st.subheader("Upload Image")
img_file_buffer = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
else:
    image = Image.open(DEMO_IMAGE)

enhancer = ImageEnhance.Brightness(image)
image = enhancer.enhance(brightness)

enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(contrast)

enhancer = ImageEnhance.Sharpness(image)
image = enhancer.enhance(sharpness)

if convert_to_grayscale:
    image = ImageOps.grayscale(image).convert("RGB")  # Convert back to RGB

if rotation_angle != 0:
    image = image.rotate(rotation_angle, expand=True)

image_np = np.array(image)

if len(image_np.shape) == 2:  
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
elif image_np.shape[-1] == 4:  
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)


st.subheader("Processed Image")
st.image(image, caption="Processed Image", use_container_width=True)


@st.cache_data
def poseDetector(frame, threshold, keypoint_color, line_color, annotate_image):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo],
                     tuple(int(line_color[i:i + 2], 16) for i in (1, 3, 5)), 3)
            cv2.ellipse(frame, points[idFrom], (5, 5), 0, 0, 360,
                        tuple(int(keypoint_color[i:i + 2], 16) for i in (1, 3, 5)), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (5, 5), 0, 0, 360,
                        tuple(int(keypoint_color[i:i + 2], 16) for i in (1, 3, 5)), cv2.FILLED)

    if annotate_image:
        for i, point in enumerate(points):
            if point:
                cv2.putText(frame, list(BODY_PARTS.keys())[i], point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return frame, points


with st.spinner("Detecting Poses..."):
    output, points = poseDetector(image_np, threshold, keypoint_color, line_color, annotate_image)

st.subheader("Pose Detection Output")
st.image(output, caption="Pose Detection", use_container_width=True)

pose_data = {"points": points}
st.subheader("Download Pose Data")
st.download_button(
    label="Download Pose Data (JSON)",
    data=json.dumps(pose_data),
    file_name="pose_data.json",
    mime="application/json"
)

# Footer (Blank Line)
st.markdown("<hr>", unsafe_allow_html=True)
