import cv2
import numpy as np
import streamlit as st

def get_exact_board_crop(img):
    """
    Locates the precise 5x5 drawn black square outline on the physical board.
    """
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to pick up dark drawn lines and corner dots
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    square_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # Look for a square covering between 10% and 75% of the total image
        if (img_w * img_h * 0.10) < area < (img_w * img_h * 0.75):
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.80 <= aspect_ratio <= 1.20:  # Must be square-like
                square_candidates.append((area, x, y, w, h))

    if square_candidates:
        # Pick the largest square contour matching the drawn boundary
        square_candidates.sort(key=lambda item: item[0], reverse=True)
        _, x, y, w, h = square_candidates[0]
        return x, y, w, h

    # Hardcoded fallback relative coordinates based on camera framing
    x = int(img_w * 0.19)
    y = int(img_h * 0.36)
    w = int(img_w * 0.50)
    h = int(img_h * 0.50)
    return x, y, w, h

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img.copy()
    img_h, img_w = img.shape[:2]

    # 1. Locate the exact 5x5 square boundary
    x, y, w, h = get_exact_board_crop(img)
    
    # Draw thick red boundary around the isolated 5x5 board
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # 2. Divide strictly into 5x5 grid cells
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    # 3. Analyze each cell using BGR & HSV color values
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Draw green cell grid
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
            
            # Center sampling core (20% padding)
            pad_x = int(cell_w * 0.20)
            pad_y = int(cell_h * 0.20)
            
            y1 = max(0, start_y + pad_y)
            y2 = min(img_h, end_y - pad_y)
            x1 = max(0, start_x + pad_x)
            x2 = min(img_w, end_x - pad_x)
            
            cell_bgr = img[y1:y2, x1:x2]
            
            if cell_bgr.size == 0:
                continue

            # Average BGR color values inside the hole core
            avg_b = np.mean(cell_bgr[:, :, 0])
            avg_g = np.mean(cell_bgr[:, :, 1])
            avg_r = np.mean(cell_bgr[:, :, 2])

            status = "no symbol"
            text_color = (160, 160, 160) # Bold Grey for Empty
            
            # --- STRICT BGR COLOR DISCRIMINATION ---
            # Red 3D Token: Red channel dominant over Blue and Green
            if avg_r > (avg_g * 1.35) and avg_r > (avg_b * 1.35) and avg_r > 70:
                status = "Red"
                text_color = (0, 0, 255) # Bright Red
                count_red += 1
            # Blue 3D Token: Blue channel dominant over Red and Green
            elif avg_b > (avg_r * 1.25) and avg_b > (avg_g * 1.10) and avg_b > 60:
                status = "Blue"
                text_color = (255, 120, 0) # Bright Blue
                count_blue += 1
            else:
                status = "no symbol"
                count_empty += 1

            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            # --- LARGE, BOLD TEXT OVERLAY ---
            display_text = "EMPTY" if status == "no symbol" else status
            
            # Font size dynamically scaled to cell size
            font_scale = max(0.8, cell_w / 90.0)
            thickness = 3
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cx = start_x + (cell_w - text_size[0]) // 2
            cy = start_y + (cell_h + text_size[1]) // 2
            
            cv2.putText(output_img, display_text, (cx, cy), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness)
            
    return output_img, status_logs, count_red, count_blue, count_empty

# --- Streamlit Presentation Layer ---
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
        st.image(processed_image_rgb, use_container_width=True, caption="5x5 Matrix Grid Scan")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
