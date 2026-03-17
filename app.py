import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Crack Detection System", layout="wide")

st.title("🚧 Crack Detection & Haul Road Monitoring")

# ==============================
# Crack Detection Function
# ==============================
def crack_density(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 130)
    
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    
    filtered = np.zeros_like(clean)
    
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        aspect_ratio = max(width/(height+1), height/(width+1))
        
        if area > 15 and aspect_ratio > 1.5:
            filtered[labels == i] = 255
    
    if np.sum(filtered > 0) < 30:
        filtered = edges
    
    density = np.sum(filtered > 0) / filtered.size
    return density, filtered

# ==============================
# Severity
# ==============================
def classify_severity(d):
    if d < 0.02:
        return "Minor"
    elif d < 0.05:
        return "Moderate"
    else:
        return "Severe"

# ==============================
# Overlay
# ==============================
def overlay(image, mask):
    img = np.array(image)
    overlay = img.copy()
    overlay[mask > 0] = [255, 0, 0]
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# ==============================
# MODE SELECTOR
# ==============================
mode = st.sidebar.radio("Select Mode", ["Single Image", "Compare Images"])

# ==============================
# SINGLE IMAGE MODE
# ==============================
if mode == "Single Image":
    st.header("📸 Single Image Analysis")
    
    file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if file:
        image = Image.open(file)
        st.image(image, use_container_width=True)
        
        density, mask = crack_density(image)
        severity = classify_severity(density)
        
        if severity == "Severe":
            risk = "HIGH RISK"
        elif severity == "Moderate":
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"
        
        overlay_img = overlay(image, mask)
        
        st.subheader("🔍 Detection")
        st.image(mask, use_container_width=True)
        
        st.subheader("🔴 Overlay")
        st.image(overlay_img, use_container_width=True)
        
        st.write(f"Density: {density:.4f}")
        st.write(f"Severity: {severity}")
        st.write(f"Risk: {risk}")

# ==============================
# COMPARE MODE
# ==============================
else:
    st.header("📊 Compare Two Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Upload Image 1", type=["jpg", "png"])
    
    with col2:
        file2 = st.file_uploader("Upload Image 2", type=["jpg", "png"])
    
    if file1 and file2:
        img1 = Image.open(file1)
        img2 = Image.open(file2)
        
        st.image([img1, img2], caption=["Image 1", "Image 2"], use_container_width=True)
        
        d1, m1 = crack_density(img1)
        d2, m2 = crack_density(img2)
        
        growth = d2 - d1
        
        sev1 = classify_severity(d1)
        sev2 = classify_severity(d2)
        
        if sev2 == "Severe" and growth > 0.02:
            risk = "CRITICAL ⚠️"
        elif sev2 == "Severe":
            risk = "HIGH RISK"
        elif sev2 == "Moderate":
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"
        
        overlay1 = overlay(img1, m1)
        overlay2 = overlay(img2, m2)
        
        st.subheader("🔴 Overlay Results")
        st.image([overlay1, overlay2], use_container_width=True)
        
        st.subheader("📊 Analysis")
        st.write(f"Initial Severity: {sev1}")
        st.write(f"Current Severity: {sev2}")
        st.write(f"Growth: {growth:.4f}")
        st.write(f"Risk: {risk}")