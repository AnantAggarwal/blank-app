import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Initial Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit App Configuration
st.set_page_config(page_title="Posture Corrector App", layout="wide")
st.title("🧘 Real-Time Posture Corrector (MediaPipe & Streamlit)")
st.markdown("This app uses your webcam to monitor your shoulder and neck posture based on a 30-frame calibration.")

# Placeholder for Streamlit session state initialization
if 'is_calibrated' not in st.session_state:
    st.session_state.is_calibrated = False
    st.session_state.calibration_frames = 0
    st.session_state.calibration_shoulder_angles = []
    st.session_state.calibration_neck_angles = []
    st.session_state.shoulder_threshold = 0
    st.session_state.neck_threshold = 0
    st.session_state.last_alert_time = 0.0
    st.session_state.alert_cooldown = 5.0 # seconds cooldown for alert

# --- Helper Functions (Defined outside the class for use with @st.cache_resource) ---

def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) between three points."""
    p1 = np.array(p1)  # First point (e.g., ear)
    p2 = np.array(p2)  # Middle point (e.g., shoulder)
    p3 = np.array(p3)  # Third point (e.g., reference point above shoulder)

    # Use screen coordinates as p1, p2, p3 for angle calculation
    # Vector from p2 to p1
    vec1 = p1 - p2
    # Vector from p2 to p3
    vec2 = p3 - p2

    # Dot product
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(np.clip(dot_product / (mag1 * mag2), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_angle(image, p1, p2, p3, angle, color, text_offset_x=20):
    """Draws points, lines, and angle text on the image (for visualization)."""
    cv2.line(image, p1, p2, color, 3)
    cv2.line(image, p2, p3, color, 3)
    cv2.circle(image, p1, 5, color, cv2.FILLED)
    cv2.circle(image, p2, 5, color, cv2.FILLED)
    cv2.circle(image, p3, 5, color, cv2.FILLED)
    
    # Calculate text position near p2
    text_pos = (p2[0] + text_offset_x, p2[1])
    cv2.putText(image, f"{angle:.1f}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# --- Streamlit-WebRTC Video Transformer ---

class PoseCorrection(VideoTransformerBase):
    """
    A VideoTransformer to perform MediaPipe Pose estimation and posture correction logic.
    """
    
    def __init__(self):
        # Initialize MediaPipe Pose *within* the transformer to avoid sharing across threads/sessions
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Convert the frame to a NumPy array (BGR format for OpenCV)
        image = frame.to_ndarray(format="bgr24")
        
        # Mirror the image for a more natural webcam feel (optional)
        image = cv2.flip(image, 1)

        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Output message/status for Streamlit display
        status_message = "Waiting for Pose..."
        status_color = (128, 128, 128) # Grey
        alert_sound_trigger = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_height, frame_width, _ = image.shape

            # STEP 2: Pose Detection - Extract key body landmarks (normalized to pixel coords)
            def _get_landmark_coords(landmark):
                return (int(landmarks[landmark.value].x * frame_width),
                        int(landmarks[landmark.value].y * frame_height))

            left_shoulder = _get_landmark_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = _get_landmark_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_ear = _get_landmark_coords(mp_pose.PoseLandmark.LEFT_EAR)
            right_ear = _get_landmark_coords(mp_pose.PoseLandmark.RIGHT_EAR)

            # Define a reference point for angle calculation (straight up from shoulder)
            ref_point = (left_shoulder[0], 0)

            # STEP 3: Angle Calculation
            # Shoulder angle: Angle between left shoulder, right shoulder, and a point vertically aligned with right shoulder
            # This is a bit non-standard; let's simplify or use a better point.
            # Assuming you want to check if shoulders are level (angle with horizontal)
            # A more robust check for a vertical reference might be better for neck/shoulder line:
            mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) // 2
            mid_shoulder = (mid_shoulder_x, left_shoulder[1]) 
            
            # Simplified: Use a horizontal reference for shoulder tilt
            shoulder_ref_horizontal = (left_shoulder[0] + 100, left_shoulder[1])
            shoulder_angle = calculate_angle(right_shoulder, left_shoulder, shoulder_ref_horizontal)
            
            # Neck angle: Angle between ear, shoulder, and a vertical reference point
            neck_angle = calculate_angle(left_ear, left_shoulder, ref_point)

            # Draw skeleton and angles
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Visualization for Shoulder Angle (as measured by the simplified method)
            draw_angle(image, right_shoulder, left_shoulder, shoulder_ref_horizontal, shoulder_angle, (255, 0, 0), text_offset_x=-100)
            
            # Visualization for Neck Angle
            draw_angle(image, left_ear, left_shoulder, ref_point, neck_angle, (0, 255, 0), text_offset_x=20)


            # STEP 1: Calibration (Handles state using Streamlit's session_state)
            if not st.session_state.is_calibrated:
                st.session_state.calibration_shoulder_angles.append(shoulder_angle)
                st.session_state.calibration_neck_angles.append(neck_angle)
                st.session_state.calibration_frames += 1
                
                status_message = f"Calibrating... {st.session_state.calibration_frames}/30"
                status_color = (0, 255, 255) # Yellow

                if st.session_state.calibration_frames >= 30:
                    st.session_state.shoulder_threshold = np.mean(st.session_state.calibration_shoulder_angles) * 0.95  # 5% margin
                    st.session_state.neck_threshold = np.mean(st.session_state.calibration_neck_angles) * 0.95 # 5% margin
                    st.session_state.is_calibrated = True
                    # Reset calibration storage after calculation (good practice)
                    st.session_state.calibration_shoulder_angles = []
                    st.session_state.calibration_neck_angles = []
            
            # STEP 4: Feedback
            if st.session_state.is_calibrated:
                current_time = time.time()
                
                # Check for bad posture (angle is below the calibrated threshold)
                if shoulder_angle < st.session_state.shoulder_threshold or neck_angle < st.session_state.neck_threshold:
                    status_message = "Poor Posture! 🚨"
                    status_color = (0, 0, 255) # Red
                    
                    if current_time - st.session_state.last_alert_time > st.session_state.alert_cooldown:
                        alert_sound_trigger = True
                        st.session_state.last_alert_time = current_time
                else:
                    status_message = "Good Posture 👍"
                    status_color = (0, 255, 0) # Green

                # Display angles/thresholds
                cv2.putText(image, f"Shoulder: {shoulder_angle:.1f}/{st.session_state.shoulder_threshold:.1f}", (10, frame_height - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Neck: {neck_angle:.1f}/{st.session_state.neck_threshold:.1f}", (10, frame_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display main status message
        cv2.putText(image, status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
        # Store alert trigger in session state for Streamlit to handle the sound (outside the thread)
        st.session_state.alert_trigger = alert_sound_trigger
        
        # Return the processed frame
        return image.tobytes()


# --- Streamlit Main Loop ---

# Streamlit-webrtc component to embed the live video stream
webrtc_ctx = webrtc_streamer(
    key="posture-corrector",
    video_processor_factory=PoseCorrection,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Sound Alert (Streamlit runs the script from top-to-bottom on state change) ---

# NOTE: playsound is NOT used. We use st.audio which requires a file path or bytes.
# We'll use a placeholder/dummy audio approach for an alert sound.
# A real implementation would require the user to provide an audio file and upload it, 
# or use a base64-encoded audio directly in HTML via st.markdown.

if st.session_state.is_calibrated:
    st.subheader("Thresholds")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Shoulder Angle Threshold", f"{st.session_state.shoulder_threshold:.1f}° (Min)")
    with col2:
        st.metric("Neck Angle Threshold", f"{st.session_state.neck_threshold:.1f}° (Min)")
        
    st.markdown("---")

    if st.session_state.is_calibrated and st.session_state.alert_trigger:
        # Simple method to play a sound: You need to have an audio file (e.g., 'alert.wav') 
        # deployed alongside your Streamlit app for this to work.
        # This will show a tiny player, but it's the safest way to trigger sound in Streamlit.
        # NOTE: Autoplay of audio without user interaction is generally blocked by browsers.
        # The user may need to interact with the page once for the audio to play correctly.
        st.warning("Poor Posture Detected! Please correct your sitting position.")
        # Replace 'alert.wav' with the path to your actual sound file.
        # Since we don't have the file, this is a placeholder demonstration.
        # st.audio("alert.wav", format="audio/wav", autoplay=True) 
        
        # To avoid re-triggering the alert sound on every rerun, reset the trigger
        st.session_state.alert_trigger = False
        
        # Alternative: Play sound using base64 encoded HTML (more complex, but better for autoplay)
        # This is more advanced and requires knowing the audio file content.

st.markdown("---")
st.info("Calibration takes **30 frames** (a few seconds) at the beginning. Sit in your desired 'Good Posture' during calibration.")