import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="bnxGkARcQkoHG64jxvtf"
)

# Treatment dictionary
treatment_plan = {
    "Gingivitis": "Maintain good oral hygiene, brush twice a day, use antiseptic mouthwash, and get professional dental cleanings regularly.",
    "Hipodonsia": "Treatment options include dental implants, bridges, or orthodontic solutions to close gaps or replace missing teeth.",
    "Kalkulus": "Dental scaling and root planing by a dentist are essential to remove tartar buildup.",
    "Kanker": "Oral cancer treatment may include surgery, radiation therapy, or chemotherapy depending on severity. Early diagnosis is crucial.",
    "Karies": "Remove decayed parts and restore with dental fillings, crowns, or root canal if severe. Prevent with fluoride and diet control.",
    "Perubahan-Warna": "Professional teeth cleaning, whitening treatments, or veneers can help. Avoid stain-causing foods and drinks.",
    "Sariawan": "Apply topical gels, rinse with salt water or antiseptic mouthwash. Avoid spicy foods and manage stress.",
    "Warna": "If discoloration is natural, no treatment needed. If pathological, consult for cleaning or aesthetic correction."
}

# Custom CSS for marquee and chat styling
st.markdown("""
    <style>
    .marquee {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        box-sizing: border-box;
        animation: marquee 12s linear infinite;
        font-size: 36px;
        font-weight: bold;
        color: white;
        background-color: cyan;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
    }

    @keyframes marquee {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    .chat-bubble {
        background-color: black;
        color: white;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
    }

    .user {
        text-align: right;
    }

    .bot {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Marquee title
st.markdown('<div class="marquee">Disease Detector</div>', unsafe_allow_html=True)
st.write("")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Chat UI
if uploaded_file is not None:
    st.markdown('<div class="chat-bubble user">📤 You uploaded:</div>', unsafe_allow_html=True)
    st.image(uploaded_file, width=300)

    # Read and encode the image
    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Run inference
    result = CLIENT.infer(encoded_image, model_id="mouthdity-classification/1")

    if result and "predicted_classes" in result and result["predicted_classes"]:
        predicted_class = result["predicted_classes"][0]
        output_img = result.get("image", "")

        st.markdown(f'<div class="chat-bubble bot">🤖 Prediction: <strong>{predicted_class}</strong></div>', unsafe_allow_html=True)

        # Display image from model if it's a base64 image
        if isinstance(output_img, str) and output_img.startswith("data:image"):
            st.image(output_img, caption="Model Prediction", use_column_width=True)

        # Show treatment plan
        treatment = treatment_plan.get(predicted_class, "Please consult a dental professional for further advice.")
        st.markdown(f'<div class="chat-bubble bot">🩺 Treatment Plan: {treatment}</div>', unsafe_allow_html=True)

        # Final message and appointment button
        st.markdown('<div class="chat-bubble bot">📍 Kindly visit <strong>DPU Dental Pimpri</strong> for more clarity on your disease.</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-bubble bot">📅 You can book your appointment below:</div>', unsafe_allow_html=True)

        st.link_button("🔗 Book Appointment Now", "https://dpuhospital.com/request-an-appointment/")

    else:
        st.markdown('<div class="chat-bubble bot">⚠️ No prediction found.</div>', unsafe_allow_html=True)
