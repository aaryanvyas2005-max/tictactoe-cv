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
    
    # 1. Edge-Based Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(edges, kernel, iterations=1)

    # 2. Outer Board Boundary Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = 0, 0, img_w, img_h
    if contours:
        board_contour = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(board_contour)
        
        # Security check: if boundary covers significant area, frame it
        if (cw * ch) > (img_w * img_h * 0.40):
            x, y, w, h = cx, cy, cw, ch
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (235, 94, 40), 5)

    # Calculate 5x5 cell dimensions
    GRID_SIZE = 5
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    
    status_logs = []
    count_x, count_o, count_empty = 0, 0, 0
    
    # First Pass: Evaluate pixel densities across all 25 cells
    cell_densities = []
    cell_data = []
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Pad cell edges by 8% to clear grid border lines
            pad_x = max(3, int(cell_w * 0.08))
            pad_y = max(3, int(cell_h * 0.08))
            
            y1 = max(0, start_y + pad_y)
            y2 = min(img_h, end_y - pad_y)
            x1 = max(0, start_x + pad_x)
            x2 = min(img_w, end_x - pad_x)
            
            cell_thresh = thresh[y1:y2, x1:x2]
            white_pixels = cv2.countNonZero(cell_thresh)
            
            cell_densities.append(white_pixels)
            cell_data.append(((row, col), start_x, end_x, start_y, end_y, cell_thresh))

    max_density = max(cell_densities) if cell_densities else 1
    
    # 3. Process 5x5 Coordinate Matrix
    for item in cell_data:
        (row, col), start_x, end_x, start_y, end_y, cell_thresh = item
        white_pixels = cv2.countNonZero(cell_thresh)
        
        # Draw green grid lines around each cell
        cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (34, 139, 34), 3)
        
        status = "no symbol"
        text_color = (140, 140, 140) # Grey for Empty
        
        # Relative noise filter for empty spaces
        if white_pixels < (max_density * 0.18) or white_pixels < 25:
            status = "no symbol"
            count_empty += 1
        else:
            cell_contours, _ = cv2.findContours(cell_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(cell_contours) > 0:
                target_contour = max(cell_contours, key=cv2.contourArea)
                
                hull = cv2.convexHull(target_contour)
                hull_area = cv2.contourArea(hull)
                solidity = cv2.contourArea(target_contour) / hull_area if hull_area > 0 else 0
                
                _, _, cw, ch = cv2.boundingRect(target_contour)
                aspect_ratio = float(cw) / ch if ch > 0 else 1.0
                
                # Shape classification
                if solidity > 0.60 and (0.70 <= aspect_ratio <= 1.40):
                    status = "O"
                    text_color = (255, 110, 0) # Orange/Blue
                    count_o += 1
                else:
                    status = "X"
                    text_color = (0, 0, 255) # Red
                    count_x += 1
            else:
                count_empty += 1
        
        status_logs.append(f"At position ({row},{col}) there is {status}")
        
        # --- LARGE, BOLD TEXT OVERLAY ---
        display_text = "Empty" if status == "no symbol" else status
        
        # Scaled font size (1.0) and thickness (3) for high visibility
        font_scale = max(0.8, cell_w / 80.0)
        text_thickness = max(2, int(font_scale * 2.5))
        
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness)[0]
        
        center_x = start_x + (cell_w - text_size[0]) // 2
        center_y = start_y + (cell_h + text_size[1]) // 2
        
        cv2.putText(output_img, display_text, (center_x, center_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, text_thickness)
            
    return output_img, status_logs, count_x, count_o, count_empty

# --- Streamlit Presentation Interface ---
st.set_page_config(page_title="5x5 CV Board Scanner Pro", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    h1 {color: #2b2d42; font-family: 'Helvetica Neue', sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 5x5 Computer Vision Tic-Tac-Toe Analyzer")
st.write("An advanced image processing engine mapping $5 \\times 5$ matrices using edge-based structural validation.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload a sharp 5x5 board image snapshot:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting a 5x5 image upload to process structural layouts.")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    processed_image, logs, x_total, o_total, empty_total = process_and_draw_board(img_bytes)
    
    with col1:
        st.success("Analysis Complete!")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Detected X's", x_total)
        m_col2.metric("Detected O's", o_total)
        m_col3.metric("Empty Spaces", empty_total)

with col2:
    if uploaded_file is not None and processed_image is not None:
        st.subheader("🖼️ Computer Vision Target Matrix (5x5)")
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="OpenCV Target Array Scan Map")
        
        st.subheader("📋 Parsed Coordinate Logs (0 to 4)")
        st.code("\n".join(logs), language="text")
