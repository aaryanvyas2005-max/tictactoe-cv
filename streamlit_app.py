import cv2
import numpy as np
import streamlit as st

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img.copy()
    img_h, img_w = img.shape[:2]
    
    # 1. Convert image to HSV Color Space for robust color isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Find the 5x5 Grid Square Boundary
    # We use grayscale thresholding to find the drawn square outline containing the 25 holes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = 0, 0, img_w, img_h
    if contours:
        # Look for a large square-like contour
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.15)]
        if valid_contours:
            board_contour = max(valid_contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(board_contour)
            x, y, w, h = cx, cy, cw, ch

    # Draw main grid border
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (235, 94, 40), 4)

    # Calculate 5x5 Cell Dimensions
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    # 3. Process each of the 25 Cells using HSV Color Masks
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Draw cell box on output
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (34, 139, 34), 2)
            
            # Crop the center 60% of the cell to avoid grid boundaries
            pad_x = int(cell_w * 0.20)
            pad_y = int(cell_h * 0.20)
            
            y1 = max(0, start_y + pad_y)
            y2 = min(img_h, end_y - pad_y)
            x1 = max(0, start_x + pad_x)
            x2 = min(img_w, end_x - pad_x)
            
            cell_hsv = hsv[y1:y2, x1:x2]
            
            if cell_hsv.size == 0:
                continue

            # --- COLOR RANGE DEFINITIONS (HSV) ---
            # Red spans across the HSV 0/180 degree wrap
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue Range
            lower_blue = np.array([90, 70, 50])
            upper_blue = np.array([135, 255, 255])

            # Generate Color Masks
            mask_red1 = cv2.inRange(cell_hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(cell_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_blue = cv2.inRange(cell_hsv, lower_blue, upper_blue)

            # Count colored pixels inside cell core
            red_pixels = cv2.countNonZero(mask_red)
            blue_pixels = cv2.countNonZero(mask_blue)
            total_cell_pixels = cell_hsv.shape[0] * cell_hsv.shape[1]

            status = "no symbol"
            text_color = (140, 140, 140) # Grey for Empty
            
            # Require at least 12% of the cropped cell area to match the color
            min_color_threshold = total_cell_pixels * 0.12

            if red_pixels > min_color_threshold and red_pixels > blue_pixels:
                status = "Red"
                text_color = (0, 0, 255) # Red text in BGR
                count_red += 1
            elif blue_pixels > min_color_threshold and blue_pixels > red_pixels:
                status = "Blue"
                text_color = (255, 100, 0) # Blue text in BGR
                count_blue += 1
            else:
                status = "no symbol"
                count_empty += 1

            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            # --- OVERLAY TEXT ---
            display_text = "Empty" if status == "no symbol" else status
            
            font_scale = max(0.6, cell_w / 100.0)
            text_thickness = 2
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness)[0]
            
            center_x = start_x + (cell_w - text_size[0]) // 2
            center_y = start_y + (cell_h + text_size[1]) // 2
            
            cv2.putText(output_img, display_text, (center_x, center_y), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, text_thickness)
            
    return output_img, status_logs, count_red, count_blue, count_empty

# --- Streamlit Presentation Layer ---
st.set_page_config(page_title="5x5 Robot Board Scanner", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    h1 {color: #2b2d42; font-family: 'Helvetica Neue', sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 5x5 Color-Based Object Matrix Scanner")
st.write("An advanced computer vision pipeline for **Team Rocket** board analysis using HSV color segmentation.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload board snapshot:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting 5x5 physical board photo upload...")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    processed_image, logs, red_total, blue_total, empty_total = process_and_draw_board(img_bytes)
    
    with col1:
        st.success("Analysis Complete!")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Red Tokens", red_total)
        m_col2.metric("Blue Tokens", blue_total)
        m_col3.metric("Empty Slots", empty_total)

with col2:
    if uploaded_file is not None and processed_image is not None:
        st.subheader("🖼️ Target Array Scan Map")
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="5x5 Object Grid Classification")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
