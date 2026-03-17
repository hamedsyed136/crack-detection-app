import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Crack Detection System", layout="wide")

st.title("🚧 AI-Based Crack Detection & Haul Road Monitoring")

# ==============================
# CRACK DETECTION + GEOMETRY
# ==============================

def crack_analysis(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)

    filtered = np.zeros_like(clean)

    lengths = []
    widths = []

    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        aspect_ratio = max(width / (height + 1), height / (width + 1))

        if area > 15 and aspect_ratio > 1.5:
            filtered[labels == i] = 255

            lengths.append(max(width, height))
            widths.append(min(width, height))

    if np.sum(filtered > 0) < 30:
        filtered = edges

    density = np.sum(filtered > 0) / filtered.size

    avg_length = np.mean(lengths) if lengths else 0
    max_length = np.max(lengths) if lengths else 0
    avg_width = np.mean(widths) if widths else 0

    return density, avg_length, max_length, avg_width, filtered


# ==============================
# ADVANCED SEVERITY (6 LEVELS)
# ==============================

def classify_severity(density):
    if density < 0.01:
        return "Hairline"
    elif density < 0.02:
        return "Low"
    elif density < 0.04:
        return "Moderate"
    elif density < 0.07:
        return "High"
    elif density < 0.1:
        return "Severe"
    else:
        return "Critical"


# ==============================
# RISK MODEL
# ==============================

def risk_model(density, width, length):
    score = (density * 50) + (width * 2) + (length * 0.01)

    if score < 2:
        return "LOW RISK", score
    elif score < 5:
        return "MEDIUM RISK", score
    elif score < 8:
        return "HIGH RISK", score
    else:
        return "CRITICAL ⚠️", score


# ==============================
# OVERLAY
# ==============================

def overlay(image, mask):
    img = np.array(image)
    overlay_img = img.copy()
    overlay_img[mask > 0] = [255, 0, 0]
    return cv2.addWeighted(img, 0.7, overlay_img, 0.3, 0)


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

        density, avg_len, max_len, avg_width, mask = crack_analysis(image)

        severity = classify_severity(density)
        risk, score = risk_model(density, avg_width, max_len)

        overlay_img = overlay(image, mask)

        st.subheader("🔍 Detection")
        st.image(mask, use_container_width=True)

        st.subheader("🔴 Overlay")
        st.image(overlay_img, use_container_width=True)

        st.subheader("📊 Analysis")

        col1, col2, col3 = st.columns(3)
        col1.metric("Crack Density", f"{density*100:.2f}%")
        col2.metric("Avg Width", f"{avg_width:.2f}px")
        col3.metric("Max Length", f"{max_len:.2f}px")

        st.write(f"### Severity: {severity}")
        st.write(f"### Risk: {risk}")
        st.write(f"### Risk Score: {score:.2f}/10")

        # Recommendation
        st.subheader("🛠 Recommendation")

        if risk == "CRITICAL ⚠️":
            st.error("Immediate road closure required")
        elif risk == "HIGH RISK":
            st.warning("Urgent maintenance required")
        elif risk == "MEDIUM RISK":
            st.info("Schedule maintenance")
        else:
            st.success("Stable condition")


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

        st.image([img1, img2], caption=["Before", "After"], use_container_width=True)

        d1, l1, m1, w1, mask1 = crack_analysis(img1)
        d2, l2, m2, w2, mask2 = crack_analysis(img2)

        growth = d2 - d1

        sev1 = classify_severity(d1)
        sev2 = classify_severity(d2)

        risk, score = risk_model(d2, w2, m2)

        overlay1 = overlay(img1, mask1)
        overlay2 = overlay(img2, mask2)

        st.subheader("🔴 Overlay Results")
        st.image([overlay1, overlay2], use_container_width=True)

        st.subheader("📊 Analysis")

        st.write(f"Initial Severity: {sev1}")
        st.write(f"Current Severity: {sev2}")
        st.write(f"Growth: {growth:.4f}")
        st.write(f"Risk: {risk}")
        st.write(f"Risk Score: {score:.2f}/10")

        if growth > 0.02:
            st.error("⚠️ Rapid crack propagation detected")
        elif growth > 0:
            st.warning("Crack increasing")
        else:
            st.success("Stable condition")
