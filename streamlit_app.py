import cv2
import numpy as np
import streamlit as st

def order_corner_points(pts):
    """
    Orders 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left order
    regardless of image rotation (0, 90, 180, or 270 degrees).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-Right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-Left has largest difference
    return rect

def get_warped_board(img):
    """
    Detects the 4 corner marker dots on the drawn 5x5 outline 
    and applies a Perspective Transform to align the grid perfectly.
    """
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to isolate small dark marker holes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for the small corner marker dots
    marker_centers = []
    for c in contours:
        area = cv2.contourArea(c)
        # Filter for small dot sizes relative to overall image resolution
        if (img_w * img_h * 0.0001) < area < (img_w * img_h * 0.008):
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = 4 * np.pi * (area / (peri * peri))
                if circularity > 0.55: # Circle shape check
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        marker_centers.append([cX, cY])

    # If 4 or more corner markers are found, build the exact bounding rectangle
    if len(marker_centers) >= 4:
        pts = np.array(marker_centers, dtype="float32")
        # Find the convex hull of the dots to grab the 4 extreme outer corner points
        hull = cv2.convexHull(pts)
        if len(hull) >= 4:
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.08 * peri, True)
            if len(approx) == 4:
                ordered_pts = order_corner_points(approx.reshape(4, 2))
                
                # Perform 500x500 Perspective Warp
                target_size = 500
                dst_pts = np.array([
                    [0, 0],
                    [target_size - 1, 0],
                    [target_size - 1, target_size - 1],
                    [0, target_size - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                warped = cv2.warpPerspective(img, M, (target_size, target_size))
                return warped, True

    # Fallback: Crop based on square contour fallback if markers are partially occluded
    square_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if (img_w * img_h * 0.15) < area < (img_w * img_h * 0.80):
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.80 <= aspect_ratio <= 1.20:
                square_candidates.append((area, x, y, w, h))

    if square_candidates:
        square_candidates.sort(key=lambda item: item[0], reverse=True)
        _, x, y, w, h = square_candidates[0]
        cropped = img[y:y+h, x:x+w]
        return cv2.resize(cropped, (500, 500)), False

    # Proportional fallback crop
    margin_w = int(img_w * 0.18)
    margin_h = int(img_h * 0.18)
    cropped = img[margin_h:img_h - margin_h, margin_w:img_w - margin_w]
    return cv2.resize(cropped, (500, 500)), False

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    # 1. Transform and align the 5x5 board into a clean 500x500 square canvas
    board_img, is_warped = get_warped_board(img)
    output_img = board_img.copy()
    
    board_size = 500
    GRID_SIZE = 5
    cell_dim = board_size // GRID_SIZE
    
    status_logs = []
    count_red, count_blue, count_empty = 0, 0, 0
    
    hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

    # 2. Iterate strictly across the 5x5 normalized matrix
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = col * cell_dim
            end_x = start_x + cell_dim
            start_y = row * cell_dim
            end_y = start_y + cell_dim
            
            # Draw green cell boundaries
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Crop 22% inner core to avoid border edges
            pad = int(cell_dim * 0.22)
            cell_hsv = hsv[start_y + pad:end_y - pad, start_x + pad:end_x - pad]
            
            if cell_hsv.size == 0:
                continue

            # --- COLOR SEGMENTATION (HSV) ---
            # Red token range (handles hue wrap)
            lower_red1 = np.array([0, 60, 40])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([155, 60, 40])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue token range (lowered S/V floor to pick up shadowed blue tokens)
            lower_blue = np.array([85, 45, 30])
            upper_blue = np.array([140, 255, 255])

            mask_r1 = cv2.inRange(cell_hsv, lower_red1, upper_red1)
            mask_r2 = cv2.inRange(cell_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            mask_blue = cv2.inRange(cell_hsv, lower_blue, upper_blue)

            red_pixels = cv2.countNonZero(mask_red)
            blue_pixels = cv2.countNonZero(mask_blue)
            total_pixels = cell_hsv.shape[0] * cell_hsv.shape[1]

            status = "no symbol"
            text_color = (160, 160, 160) # Grey for empty
            
            min_threshold = total_pixels * 0.12

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
            
            # --- PROMINENT, BOLD TEXT OVERLAY ---
            display_text = "EMPTY" if status == "no symbol" else status
            font_scale = 0.75
            thickness = 2
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cx = start_x + (cell_dim - text_size[0]) // 2
            cy = start_y + (cell_dim + text_size[1]) // 2
            
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
st.write("Perspective-corrected $5 \\times 5$ matrix scanner driven by corner anchor dot detection.")
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
        st.image(processed_image_rgb, use_container_width=True, caption="5x5 Normalized Board View")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
