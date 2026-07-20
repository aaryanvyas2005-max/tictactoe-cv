import cv2
import numpy as np
import streamlit as st

def detect_and_crop_grid(img):
    """
    Locates the inner 5x5 drawn square boundary regardless of image orientation.
    """
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to pick up the drawn black grid lines
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    square_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # Target square must cover between 12% and 80% of the image canvas
        if (img_w * img_h * 0.12) < area < (img_w * img_h * 0.80):
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.75 <= aspect_ratio <= 1.25: # Square profile check
                square_candidates.append((area, x, y, w, h))

    if square_candidates:
        square_candidates.sort(key=lambda item: item[0], reverse=True)
        _, x, y, w, h = square_candidates[0]
        return x, y, w, h

    # Dynamic fallback proportional to board geometry
    x = int(img_w * 0.18)
    y = int(img_h * 0.18)
    w = int(img_w * 0.64)
    h = int(img_h * 0.64)
    return x, y, w, h

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img.copy()
    img_h, img_w = img.shape[:2]

    # 1. Locate the exact 5x5 grid bounding box
    x, y, w, h = detect_and_crop_grid(img)
    
    # Draw thick red boundary around detected 5x5 board
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # 2. Divide into 5x5 grid cells
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    # Convert image to HSV for shadow-immune color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Process every cell in the 5x5 matrix
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Draw green box for cell boundaries
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
            
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
            # Red range (handles hue wrap around 0 / 180 degrees)
            lower_red1 = np.array([0, 50, 40])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([155, 50, 40])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue range (lowered V & S thresholds for shadowed lighting)
            lower_blue = np.array([85, 40, 30])
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
            
            # Requires at least 10% coverage of token color in hole core
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
            
            # --- BOLD, EXTRA-LARGE TEXT OVERLAY ---
            display_text = "EMPTY" if status == "no symbol" else status
            
            font_scale = max(0.8, cell_w / 75.0)
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
st.write("Strict $5 \\times 5$ matrix detection identifying 3D Red and Blue tokens from empty board slots.")
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
