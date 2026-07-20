import cv2
import numpy as np
import streamlit as st

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img_bgr.copy()
    img_h, img_w = img_bgr.shape[:2]

    # Convert to RGB and HSV explicitly
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Exact Bounding Box for the 5x5 grid based on the full square drawn on board
    # We expand margins slightly to include all 5 rows and 5 columns perfectly
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Precise default alignment matching the board's 5x5 grid boundary
    x, y, w, h = int(img_w * 0.185), int(img_h * 0.285), int(img_w * 0.615), int(img_h * 0.525)
    
    square_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if (img_w * img_h * 0.15) < area < (img_w * img_h * 0.75):
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            if 0.82 <= aspect_ratio <= 1.18:
                square_candidates.append((area, bx, by, bw, bh))

    if square_candidates:
        square_candidates.sort(key=lambda item: item[0], reverse=True)
        _, x, y, w, h = square_candidates[0]

    # Draw outer red frame across all 5x5 holes
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 4)

    # 2. Slice strictly into 5x5 Grid Cells
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    # 3. Process every cell
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Draw green cell grid
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Crop 25% center core of each hole
            pad_x = int(cell_w * 0.25)
            pad_y = int(cell_h * 0.25)
            
            y1 = max(0, start_y + pad_y)
            y2 = min(img_h, end_y - pad_y)
            x1 = max(0, start_x + pad_x)
            x2 = min(img_w, end_x - pad_x)
            
            cell_rgb = img_rgb[y1:y2, x1:x2]
            cell_hsv = hsv[y1:y2, x1:x2]
            
            if cell_rgb.size == 0:
                continue

            # HSV Masks for Red and Blue Tokens
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([12, 255, 255])
            lower_red2 = np.array([165, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            lower_blue = np.array([90, 60, 40])
            upper_blue = np.array([135, 255, 255])

            mask_r1 = cv2.inRange(cell_hsv, lower_red1, upper_red1)
            mask_r2 = cv2.inRange(cell_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            mask_blue = cv2.inRange(cell_hsv, lower_blue, upper_blue)

            red_pixels = cv2.countNonZero(mask_red)
            blue_pixels = cv2.countNonZero(mask_blue)
            total_pixels = cell_hsv.shape[0] * cell_hsv.shape[1]

            status = "no symbol"
            text_color = (0, 0, 255) # Red text in BGR format
            
            # Threshold: if > 12% of hole center matches Red/Blue
            min_thresh = total_pixels * 0.12

            if red_pixels > min_thresh and red_pixels >= blue_pixels:
                status = "Red"
                text_color = (0, 0, 255) # Bright Red
                count_red += 1
            elif blue_pixels > min_thresh and blue_pixels > red_pixels:
                status = "Blue"
                text_color = (255, 120, 0) # Bright Blue
                count_blue += 1
            else:
                status = "no symbol"
                text_color = (120, 120, 120)
                count_empty += 1

            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            # --- HIGH-VISIBILITY BOLD TEXT OVERLAY ---
            display_text = "EMPTY" if status == "no symbol" else status
            font_scale = 0.55
            thickness = 2
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cx = start_x + (cell_w - text_size[0]) // 2
            cy = start_y + (cell_h + text_size[1]) // 2
            
            cv2.putText(output_img, display_text, (cx, cy), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness)
            
    return output_img, status_logs, count_red, count_blue, count_empty

# --- Streamlit Presentation View ---
st.set_page_config(page_title="5x5 Robot Board Scanner", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    h1 {color: #2b2d42; font-family: 'Helvetica Neue', sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 5x5 Robot Board Matrix Scanner")
st.write("Strict $5 \\times 5$ grid classification for 3D Red and Blue tokens.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload 5x5 board snapshot:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting image upload...")

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
        st.subheader("🖼️ Target 5x5 Grid Scan Map")
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="5x5 Matrix Scan Map")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
