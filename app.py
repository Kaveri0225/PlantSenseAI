import streamlit as st
from PIL import Image
from predict import load_model

st.set_page_config(
    page_title="Plant Sense AI",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def get_classifier():
    return load_model()

st.title("🌿 Plant Sense AI")
st.subheader("AI-Powered Plant Disease Detection")
st.write("Upload a leaf image to check its possible health condition.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        st.markdown("### Leaf Image Analysis")
        st.write("Image uploaded successfully!")
        st.write("Ready for AI classification.")

    if st.button("🔍 Analyze Plant", type="primary"):
        try:
            with st.spinner("AI is analyzing the leaf image..."):
                classifier = get_classifier()
                predictions = classifier(image, top_k=3)

            best = predictions[0]
            label = best["label"].replace("___", " - ").replace("_", " ")
            confidence = best["score"]

            st.divider()
            st.header("🌱 Prediction Result")

            if confidence < 0.50:
                st.warning("Low confidence prediction. Please try a clearer image.")
            elif "healthy" in best["label"].lower():
                st.success("The model predicts this leaf is HEALTHY.")
            else:
                st.warning("The model predicts a possible disease.")

            st.metric("Predicted Class", label)
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
            st.progress(float(confidence))

            st.subheader("Top 3 Predictions")
            for item in predictions:
                readable = item["label"].replace("___", " - ").replace("_", " ")
                st.write(f"**{readable}** — {item['score'] * 100:.2f}%")

            st.info(
                "This is an AI-based prediction, not a confirmed diagnosis. "
                "Photo quality and real-world conditions may affect results. "
                "Consult an agricultural expert before treatment."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Check your internet connection and try again.")

else:
    st.info("👆 Upload a leaf image to start.")

st.divider()
st.subheader("✨ Project Features")

c1, c2, c3 = st.columns(3)

c1.markdown("### 📷 Image Upload")
c1.write("Upload a plant leaf photo.")

c2.markdown("### 🤖 AI Classification")
c2.write("Uses a pretrained deep-learning model.")

c3.markdown("### 📊 Confidence Score")
c3.write("Displays prediction confidence.")

st.caption("Plant Sense AI | Academic Project Prototype")