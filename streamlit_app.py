import cv2
import numpy as np
import streamlit as st

def process_and_draw_board(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, ["Error reading image format."], 0, 0, 0
        
    output_img = img.copy()
    
    # 1. Image Processing & Binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Boundary Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return output_img, ["No distinctive board grid found. Try a clearer picture."], 0, 0, 0

    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)
    
    # Draw Outer Border
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (235, 94, 40), 4) # Modern Coral Border
    
    cell_w = w // 3
    cell_h = h // 3
    
    status_logs = []
    count_x, count_o, count_empty = 0, 0, 0
    
    # 3. Process 3x3 Coordinate Matrix
    for row in range(3):
        for col in range(3):
            start_x = x + (col * cell_w)
            end_x = start_x + cell_w
            start_y = y + (row * cell_h)
            end_y = start_y + cell_h
            
            # Subtle green grid lines
            cv2.rectangle(output_img, (start_x, start_y), (end_x, end_y), (34, 139, 34), 2)
            
            # Inner pad slice
            pad_x = int(cell_w * 0.05)
            pad_y = int(cell_h * 0.05)
            cell_thresh = thresh[start_y + pad_y:end_y - pad_y, start_x + pad_x:end_x - pad_x]
            
            cell_contours, hierarchy = cv2.findContours(cell_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in cell_contours if cv2.contourArea(c) > 30]
            
            status = "no symbol"
            text_color = (140, 140, 140) 
            
            if len(valid_contours) > 0:
                has_hole = False
                if hierarchy is not None:
                    for h_info in hierarchy[0]:
                        if h_info[2] != -1:
                            has_hole = True
                            break
                
                if has_hole:
                    status = "O"
                    text_color = (255, 110, 0) # Electric Blue
                    count_o += 1
                else:
                    status = "X"
                    text_color = (0, 0, 255) # Deep Red
                    count_x += 1
            else:
                count_empty += 1
            
            status_logs.append(f"At position ({row},{col}) there is {status}")
            
            # Text layout overlay
            display_text = "Empty" if status == "no symbol" else status
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
            
            center_x = start_x + (cell_w - text_size[0]) // 2
            center_y = start_y + (cell_h + text_size[1]) // 2
            cv2.putText(output_img, display_text, (center_x, center_y), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 2)
            
    return output_img, status_logs, count_x, count_o, count_empty

# --- Streamlit Presentation Config ---
st.set_page_config(page_title="CV Board Scanner Pro", page_icon="🤖", layout="wide")

# Modern Styling Clean Look
st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    .reportview-container .main .block-container{padding-top: 2rem;}
    h1 {color: #2b2d42; font-family: 'Helvetica Neue', sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Computer Vision Tic-Tac-Toe Analyzer")
st.write("An advanced image processing engine mapping $3 \\times 3$ matrices using structural contour validation.")
st.markdown("---")

# Layout distribution columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Node")
    uploaded_file = st.file_uploader("Upload a sharp board image snapshot:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("💡 Awaiting an image upload to process structural layouts.")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    processed_image, logs, x_total, o_total, empty_total = process_and_draw_board(img_bytes)
    
    with col1:
        st.success("Analysis Complete!")
        # Executive Summary Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Detected X's", x_total)
        m_col2.metric("Detected O's", o_total)
        m_col3.metric("Empty Spaces", empty_total)

    with col2:
        st.subheader("🖼️ Computer Vision Target Matrix")
        if processed_image is not None:
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, use_container_width=True, caption="OpenCV Target Array Scan Map")
            
            st.subheader("📋 Parsed Coordinate Logs")
            st.code("\n".join(logs), language="text")
