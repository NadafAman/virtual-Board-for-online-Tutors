import datetime
import textwrap
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Magic Teaching Board 🎨", 
    page_icon="🚀", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.output_text = ""
    st.session_state.selected_color = (255, 0, 255)
    st.session_state.brush_size = 7
    st.session_state.current_frame = None
    st.session_state.camera_index = 0
    st.session_state.cap = None
    st.session_state.canvas = None
    st.session_state.response_canvas = None  # For AI response display

# Add custom CSS


st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #FFE5B4 0%, #FFFFE0 100%);
        font-family: 'Comic Neue', cursive, 'Montserrat', sans-serif;
    }

    /* Vibrant Title Styling */
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: 900;
        margin-bottom: 30px;
        background: linear-gradient(45deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
        letter-spacing: 2px;
    }

    /* Playful Card Container */
    .stContainer {
        background-color: white;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 20px;
        border: 3px dashed #FFD700;
        transition: all 0.4s ease;
    }

    }

    .stButton > button {
        background: blue;
        color: white;
        border: 2px solid white;
        border-radius: 15px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 12px 25px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    }

    .answer-text {
        background: linear-gradient(to right, #FFD700 0%, #FFA500 100%);
        color: white;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        font-size: 18px;
        line-height: 1.7;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border: 3px solid white;
        font-family: 'Comic Neue', cursive;
    }

    /* Colorful Radio Buttons */
    .stRadio > div {
        background-color: #FFFACD;
        padding: 15px;
        border: 2px dashed #FF69B4;
    }
    .stRadio [data-testid='stRadioButtonContainer'] > div {
        background-color: white;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .stRadio [data-testid='stRadioButtonContainer'] > div:hover {
        border-color: #4ECDC4;
    }



    /* Camera Frame with Personality */
    .stImage {
        box-shadow: 0 12px 25px rgba(0,0,0,0.15);
        transition: all 0.4s ease;
    }

    </style>
    <link href="https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Stylish title with playful emojis
st.markdown('<h1 class="title">🎨✨ Magic Teaching Board 🤖🖌️</h1>', unsafe_allow_html=True)

# Camera selection and initialization
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
    if st.session_state.response_canvas is None:
        st.session_state.response_canvas = np.zeros_like(test_frame)
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
    st.session_state.brush_size = st.slider("Brush Size", 5, 30, 7)

with col2:
    st.markdown("### AI Response")

    # Create a placeholder for AI responses
    ai_response_box = st.empty()

    # Display the current response (if available)
    if st.session_state.output_text:
        ai_response_box.markdown(
            f'<div class="answer-text">{st.session_state.output_text}</div>', 
            unsafe_allow_html=True
        )

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
                st.session_state.response_canvas = np.zeros_like(st.session_state.canvas)
                st.session_state.output_text = ""



# Initialize AI
genai.configure(api_key="GEMINI_API_KEY")   #insert your api for this model 
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
    elif fingers == [1, 1, 0, 0, 1]:  # spidy for clearing
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)
        st.session_state.response_canvas = np.zeros_like(st.session_state.canvas)
        st.session_state.output_text = ""
    
    return current_pos

def write_multiline_text(canvas, text, pos=(50, 100), font=cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale=0.7, color=(255, 255, 255), thickness=2, line_spacing=30):
    y0, x0 = pos
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * line_spacing
        cv2.putText(canvas, line, (x0, y), font, font_scale, color, thickness, cv2.LINE_AA)

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # 4 fingers up
        try:
            pil_image = Image.fromarray(canvas)
            prompt =  """Analyze this image in detail:
                
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
                st.error("Failed to capture frame. Reinitializing...")
                st.session_state.cap.release()
                st.session_state.cap = cv2.VideoCapture(camera_index)
                continue
                
            img = cv2.flip(img, 1)
            st.session_state.response_canvas = np.zeros_like(img)
            info = getHandInfo(img)
            if info:
                fingers, lmList = info
                prev_pos = draw(info, prev_pos)
                new_text = sendToAI(model, st.session_state.canvas, fingers)
                if new_text:
                    st.session_state.output_text = new_text
                    ai_response_box.markdown(
                        f'<div class="answer-text">{st.session_state.output_text}</div>', 
                        unsafe_allow_html=True
                    )
            
            image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
            image_combined = cv2.addWeighted(image_combined, 1, st.session_state.response_canvas, 0.5, 0)
            FRAME_WINDOW.image(image_combined, channels="BGR")
            st.session_state.current_frame = image_combined
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
