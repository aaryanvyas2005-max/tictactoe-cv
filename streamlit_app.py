import cv2
import numpy as np
import streamlit as st

def detect_board_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img.shape[:2]
    
    possible_grids = []
    for c in contours:
        area = cv2.contourArea(c)
        if (img_w * img_h * 0.10) < area < (img_w * img_h * 0.85):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                possible_grids.append((area, approx))
                
    if possible_grids:
        possible_grids.sort(key=lambda x: x[0], reverse=True)
        pts = possible_grids[0][1].reshape(4, 2)
        return order_points(pts)
        
    margin_w = int(img_w * 0.15)
    margin_h = int(img_h * 0.15)
    pts = np.array([
        [margin_w, margin_h],
        [img_w - margin_w, margin_h],
        [img_w - margin_w, img_h - margin_h],
        [margin_w, img_h - margin_h]
    ], dtype="float32")
    return pts

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    corners = detect_board_corners(img)
    
    board_size = 600
    dst_pts = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))
    
    output_img = warped.copy()
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    GRID_SIZE = 5
    cell_dim = board_size // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = col * cell_dim
            end_x = start_x + cell_dim
            start_y = row * cell_dim
            end_y = start_y + cell_dim
            
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Use 15% padding so we catch tokens that sit slightly off-center
            pad = int(cell_dim * 0.15)
            cell_hsv = hsv[start_y + pad:end_y - pad, start_x + pad:end_x - pad]
            
            if cell_hsv.size == 0:
                continue

            # --- EXPANDED HSV RANGES ---
            # Red Range (Broadened for lighting variations)
            lower_red1 = np.array([0, 50, 40])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([155, 50, 40])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue Range (Lowered S and V thresholds to detect shadowed blue tokens)
            lower_blue = np.array([85, 40, 30])
            upper_blue = np.array([140, 255, 255])

            mask_r1 = cv2.inRange(cell_hsv, lower_red1, upper_red1)
            mask_r2 = cv2.inRange(cell_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            
            mask_blue = cv2.inRange(cell_hsv, lower_blue, upper_blue)

            red_count = cv2.countNonZero(mask_red)
            blue_count = cv2.countNonZero(mask_blue)
            total_pixels = cell_hsv.shape[0] * cell_hsv.shape[1]

            status = "no symbol"
            text_color = (140, 140, 140) 
            
            # Lowered threshold to 10% area coverage
            min_pixels = total_pixels * 0.10

            if red_count > min_pixels and red_count >= blue_count:
                status = "Red"
                text_color = (0, 0, 255) 
                count_red += 1
            elif blue_count > min_pixels and blue_count > red_count:
                status = "Blue"
                text_color = (255, 100, 0) 
                count_blue += 1
            else:
                status = "no symbol"
                count_empty += 1

            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            display_text = "Empty" if status == "no symbol" else status
            font_scale = 0.65
            thickness = 2
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cx = start_x + (cell_dim - text_size[0]) // 2
            cy = start_y + (cell_dim + text_size[1]) // 2
            
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

st.title("🤖 5x5 Color Token Matrix Scanner")
st.write("Strict $5 \\times 5$ grid detection isolating 3D Red and Blue tokens from empty board slots.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload 5x5 robot board photo:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting 5x5 board image upload...")

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
        st.subheader("🖼️ Cropped 5x5 Grid Scan Map")
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="Perspective-Corrected 5x5 Matrix")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
