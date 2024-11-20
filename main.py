import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector 
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import datetime
import textwrap
import time

# Page configuration
st.set_page_config(layout="wide")
st.title("Virtual Teaching Board")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.output_text = ""
    st.session_state.selected_color = (255, 0, 255)
    st.session_state.brush_size = 10
    st.session_state.current_frame = None
    st.session_state.camera_index = 0
    st.session_state.cap = None
    st.session_state.canvas = None
    st.session_state.last_request_time = 0
    st.session_state.can_send_request = True

# Add custom CSS
st.markdown("""
    <style>
    .answer-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.5;
        border: 1px solid #e0e0e0;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Camera selection and initialization code remains the same...
available_cameras = []
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

if not available_cameras:
    st.error("No cameras detected. Please connect a camera and restart the application.")
    st.stop()

camera_index = st.selectbox(
    "Select Camera",
    options=available_cameras,
    format_func=lambda x: f"Camera {x}",
    index=0 if 0 in available_cameras else 0
)

try:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = cv2.VideoCapture(camera_index)
    st.session_state.cap.set(3, 1280)
    st.session_state.cap.set(4, 720)
    
    success, test_frame = st.session_state.cap.read()
    if not success or test_frame is None:
        st.error(f"Failed to initialize camera {camera_index}. Please try another camera.")
        st.stop()
    
    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(test_frame)
    
except Exception as e:
    st.error(f"Error initializing camera: {str(e)}")
    st.stop()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox('Run Camera', value=True)
    FRAME_WINDOW = st.image([])
    
    st.markdown("### Drawing Controls")
    color_options = {
        "Magenta": (255, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Red": (255, 0, 0),
        "Yellow": (255, 255, 0)
    }
    selected_color_name = st.radio("Select Drawing Color", list(color_options.keys()))
    st.session_state.selected_color = color_options[selected_color_name]
    st.session_state.brush_size = st.slider("Brush Size", 5, 30, 10)

with col2:
    st.markdown("### AI Response")
    answer_container = st.container()
    with answer_container:
        if st.session_state.output_text:
            st.markdown(f'<div class="answer-text">{st.session_state.output_text}</div>', 
                       unsafe_allow_html=True)
    
    st.markdown("### Actions")
    col_actions1, col_actions2 = st.columns(2)
    with col_actions1:
        if st.button("📸 Take Screenshot"):
            if st.session_state.current_frame is not None:
                filename = f"board_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, st.session_state.current_frame)
                st.success(f"Screenshot saved as {filename}")
    
    with col_actions2:
        if st.button("🗑️ Clear Canvas"):
            if st.session_state.canvas is not None:
                st.session_state.canvas = np.zeros_like(st.session_state.canvas)
                st.session_state.output_text = ""

    st.markdown("""
    ### Gesture Guide
    - ☝️ Index finger up: Draw
    - 👍 Thumb up: Clear canvas
    - ✋ All fingers up: Get AI analysis (5 second cooldown)
    """)

# Initialize AI
genai.configure(api_key="AIzaSyDXEtcgE5pHEnQZ9_F0s8uDSGlIFravsV4") 
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos):
    fingers, lmList = info
    current_pos = None
    
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(st.session_state.canvas, current_pos, prev_pos, 
                st.session_state.selected_color, 
                st.session_state.brush_size)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up for clearing
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)
        st.session_state.output_text = ""
    
    return current_pos

def check_cooldown():
    current_time = time.time()
    if current_time - st.session_state.last_request_time >= 5:  # 5 second cooldown
        st.session_state.can_send_request = True
        return True
    return False

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # All fingers up
        if not check_cooldown():
            return None
        
        if st.session_state.can_send_request:
            st.session_state.can_send_request = False  # Prevent new requests
            st.session_state.last_request_time = time.time()  # Update last request time
            st.session_state.output_text = ""  # Clear previous response
            
            try:
                pil_image = Image.fromarray(canvas)
                prompt = """Analyze this image in detail:
                
1. If it's a mathematical problem:
   - Show the complete step-by-step solution
   - Explain each step clearly
   - Provide the final answer
   - Point out any important concepts used

2. If it's text or writing:
   - Interpret the content accurately
   - Provide relevant explanations
   - Add helpful context or examples
   - Suggest related topics if relevant

3. If it's a diagram or drawing:
   - Describe what you see in detail
   - Explain the key components
   - Discuss the relationships shown
   - Provide relevant context

4. If it's a combination:
   - Address each component separately
   - Show how they relate to each other
   - Provide a comprehensive analysis"""

                response = model.generate_content([prompt, pil_image])
                return textwrap.fill(response.text, width=60)
            except Exception as e:
                return f"Error processing image: {str(e)}"
    return None

# Main loop variables
prev_pos = None

# Main application loop
if run:
    try:
        while True:
            if not st.session_state.cap.isOpened():
                st.error("Camera connection lost. Please restart the application.")
                break

            success, img = st.session_state.cap.read()
            if not success or img is None:
                st.error("Failed to capture frame. Trying to reinitialize camera...")
                st.session_state.cap.release()
                st.session_state.cap = cv2.VideoCapture(camera_index)
                continue
                
            img = cv2.flip(img, 1)
            
            info = getHandInfo(img)
            if info:
                fingers, lmList = info
                prev_pos = draw(info, prev_pos)
                new_text = sendToAI(model, st.session_state.canvas, fingers)
                if new_text:
                    st.session_state.output_text = new_text
            
            image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
            FRAME_WINDOW.image(image_combined, channels="BGR")
            st.session_state.current_frame = image_combined
            
            if st.session_state.output_text:
                with answer_container:
                    st.markdown(f'<div class="answer-text">{st.session_state.output_text}</div>', 
                              unsafe_allow_html=True)
            
            cv2.waitKey(1)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        if st.session_state.cap is not None:
            st.session_state.cap.release()

else:
    st.warning("Click 'Run Camera' to start the virtual board.")

if st.button("Stop Application"):
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.experimental_rerun()