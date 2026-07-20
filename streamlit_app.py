import cv2
import numpy as np
import streamlit as st

def process_and_draw_board(image_bytes, crop_x, crop_y, crop_w, crop_h):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img.copy()
    img_h, img_w = img.shape[:2]

    # Convert percentages to pixel values for the yellow highlighted 5x5 square
    x = int((crop_x / 100.0) * img_w)
    y = int((crop_y / 100.0) * img_h)
    w = int((crop_w / 100.0) * img_w)
    h = int((crop_h / 100.0) * img_h)

    # Draw the main bounding box around the 5x5 grid in bright red
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 4)

    # Divide strictly into a 5x5 Grid
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Process all 25 cells inside the highlighted grid
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Draw green cell boundaries
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Crop center core (20% padding) to sample token top surfaces cleanly
            pad_x = int(cell_w * 0.20)
            pad_y = int(cell_h * 0.20)
            
            y1 = max(0, start_y + pad_y)
            y2 = min(img_h, end_y - pad_y)
            x1 = max(0, start_x + pad_x)
            x2 = min(img_w, end_x - pad_x)
            
            cell_hsv = hsv[y1:y2, x1:x2]
            
            if cell_hsv.size == 0:
                continue

            # --- COLOR RANGES IN HSV ---
            # Red token range
            lower_red1 = np.array([0, 50, 40])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([155, 50, 40])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue token range
            lower_blue = np.array([85, 40, 25])
            upper_blue = np.array([140, 255, 255])

            mask_r1 = cv2.inRange(cell_hsv, lower_red1, upper_red1)
            mask_r2 = cv2.inRange(cell_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            mask_blue = cv2.inRange(cell_hsv, lower_blue, upper_blue)

            red_pixels = cv2.countNonZero(mask_red)
            blue_pixels = cv2.countNonZero(mask_blue)
            total_pixels = cell_hsv.shape[0] * cell_hsv.shape[1]

            status = "no symbol"
            text_color = (180, 180, 180) # Grey for Empty
            
            min_threshold = total_pixels * 0.10

            if red_pixels > min_threshold and red_pixels >= blue_pixels:
                status = "Red"
                text_color = (0, 0, 255) # Bright Red
                count_red += 1
            elif blue_pixels > min_threshold and blue_pixels > red_pixels:
                status = "Blue"
                text_color = (255, 120, 0) # Bright Blue
                count_blue += 1
            else:
                status = "no symbol"
                count_empty += 1

            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            # --- BOLD, LARGE OVERLAY TEXT ---
            display_text = "EMPTY" if status == "no symbol" else status
            font_scale = max(0.7, cell_w / 70.0)
            thickness = 3
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cx = start_x + (cell_w - text_size[0]) // 2
            cy = start_y + (cell_h + text_size[1]) // 2
            
            cv2.putText(output_img, display_text, (cx, cy), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness)
            
    return output_img, status_logs, count_red, count_blue, count_empty

# --- Streamlit Layout ---
st.set_page_config(page_title="5x5 Robot Board Scanner", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    h1 {color: #2b2d42; font-family: 'Helvetica Neue', sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 5x5 Robot Board Matrix Scanner")
st.write("Strict $5 \\times 5$ matrix detection using manual/calibrated grid alignment.")
st.markdown("---")

# Calibration Sliders (Presets calibrated to match your photo exactly)
with st.sidebar:
    st.header("🎯 Grid Calibration")
    st.write("Adjust these sliders if your camera angle shifts:")
    crop_x = st.slider("Grid Left Position (%)", 0, 50, 18)
    crop_y = st.slider("Grid Top Position (%)", 0, 50, 6)
    crop_w = st.slider("Grid Width (%)", 30, 90, 50)
    crop_h = st.slider("Grid Height (%)", 30, 90, 72)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload 5x5 board snapshot:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting image upload...")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    processed_image, logs, red_total, blue_total, empty_total = process_and_draw_board(
        img_bytes, crop_x, crop_y, crop_w, crop_h
    )
    
    with col1:
        st.success("Analysis Complete!")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Red Tokens", red_total)
        m_col2.metric("Blue Tokens", blue_total)
        m_col3.metric("Empty Slots", empty_total)

with col2:
    if uploaded_file is not None and processed_image is not None:
        st.subheader("🖼️ Target 5x5 Grid Scan Map")
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="Calibrated 5x5 Matrix Scan View")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
