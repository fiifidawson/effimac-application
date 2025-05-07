# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# import io

# # Set page configuration
# st.set_page_config(
#     page_title="RetinaDX-NN: Alzheimer's Prediction",
#     page_icon="🧠",
#     layout="wide"
# )

# # Add title and description
# st.title("RetinaDX-NN: Alzheimer's Prediction from Retinal Images")
# st.markdown("""
# This application uses a deep learning model to predict Alzheimer's disease risk based on retinal images.
# Upload a retinal image to get a prediction and visualization of the model's attention.
# """)

# # Define class labels
# class_labels = {
#     0: ("CN (Cognitively Normal)", "low_risk"),
#     1: ("MCI (Mild Cognitive Impairment)", "medium_risk"),
#     2: ("AD (Alzheimer's Disease)", "high_risk")
# }

# # Class mapping
# idx_to_class = {
#     0: "CN (Cognitively Normal)",
#     1: "MCI (Mild Cognitive Impairment)",
#     2: "AD (Alzheimer's Disease)",
# }

# # Extended explanations for each class
# extended_explanations = {
#     "CN (Cognitively Normal)": (
#         "\nThe model did not find significant abnormalities in the retinal structure, leading to a normal classification. "
#         "\nThe absence of heatmap intensity in critical regions suggests no concerning signs of disease. "
#         "\nA well-defined and evenly structured retina supports this assessment."
#         "\n\nNext Step: Routine eye exams are still recommended for continued eye health."
#     ),
#     "MCI (Mild Cognitive Impairment)": (
#         "\nThe model focused on bright, distinct deposits beneath the retina, which are characteristic of Drusen. "
#         "\nThese deposits, often found near the macula, can contribute to vision impairment if they grow larger. "
#         "\nThe heatmap highlights these abnormal deposits, reinforcing the likelihood of this condition."
#         "\n\nNext Step: Regular OCT scans are advised to monitor Drusen size and density."
#     ),
#     "AD (Alzheimer's Disease)": (
#         "\nCNV (Choroidal Neovascularization)"
#         "\nThe model highlighted abnormal vascular regions, often associated with excessive blood vessel growth. "
#         "\nThese regions may indicate leakage or neovascularization, commonly seen in wet AMD. "
#         "\nThe heatmap shows the AI's focus on irregular patterns in the retina, which aligns with CNV characteristics."
#         "\n\nNext Step: Confirm with Fluorescein Angiography or OCT Angiography to assess neovascularization."
#         "\n\nDME (Diabetic Macular Edema)"
#         "\nThe model detected fluid accumulation in the macula, emphasizing regions with potential swelling. "
#         "\nThe highlighted areas suggest changes in retinal thickness, which are key indicators of macular edema. "
#         "\nThe intensity of the heatmap in the central macular zone supports this diagnosis."
#         "\n\nNext Step: Confirm with Fundus Photography or Additional OCT scans to evaluate macular thickness."
#     )
# }

# # Heatmap color interpretation
# heatmap_explanation = (
#     "\nRed areas indicate the most critical regions influencing the AI's decision, suggesting high abnormality. "
#     "\nOrange and yellow areas represent moderate attention, possibly indicating early signs of disease. "
#     "\nBlue and green areas contribute the least to the decision, implying normal or less concerning regions."
# )

# def find_last_conv_layer(model):
#     """
#     Find the name of the last convolutional-type layer (Conv2D or DepthwiseConv2D) in a Keras model.
#     """
#     from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
    
#     # Iterate layers in reverse so we find the last one first
#     for layer in reversed(model.layers):
#         # Check if this layer is a standard or depthwise convolution
#         if isinstance(layer, (Conv2D, DepthwiseConv2D)):
#             return layer.name
    
#     # If loop completes without returning, no conv-type layer was found
#     raise ValueError("No Conv2D or DepthwiseConv2D layer found in the model.")

# def grad_cam(model, img_array, layer_name):
#     """Apply Grad-CAM technique to visualize model's attention"""
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         predicted_class = tf.argmax(predictions[0])  # Get the predicted class index
#         loss = predictions[:, predicted_class]

#     grads = tape.gradient(loss, conv_outputs)  # Compute gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)  # Weighted activation

#     # Normalize heatmap
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

#     return heatmap, predicted_class.numpy()

# def overlay_heatmap(img_array, heatmap, alpha=0.4):
#     """Overlay Grad-CAM heatmap on the original image"""
#     heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))  # Resize heatmap
#     heatmap = np.uint8(255 * heatmap)  # Convert to RGB format
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

#     img = np.uint8(255 * img_array[0])  # Convert image to correct format
#     overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Merge heatmap with image
    
#     # Convert from BGR (OpenCV default) to RGB for matplotlib/streamlit
#     overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    
#     return overlayed_img

# def process_image(uploaded_file, model):
#     # Open and preprocess image
#     img = Image.open(uploaded_file).convert('RGB')
#     # Resize to the expected input shape of the model (300, 300) and keep RGB channels
#     img = img.resize((300, 300))
#     img_array = np.array(img).astype('float32') / 255.0
#     # Add batch dimension
#     img_array = img_array.reshape((1, 300, 300, 3))
    
#     # Get the last convolutional layer
#     last_conv_layer_name = find_last_conv_layer(model)
    
#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class_idx = np.argmax(predictions[0])
#     confidence = predictions[0][predicted_class_idx] * 100
    
#     # Get class name and risk level
#     class_name, risk_level = class_labels[predicted_class_idx]
    
#     # Generate Grad-CAM visualization
#     heatmap, _ = grad_cam(model, img_array, layer_name=last_conv_layer_name)
#     overlayed_img = overlay_heatmap(img_array, heatmap)
    
#     # Get extended explanation
#     predicted_class_name = idx_to_class[predicted_class_idx]
#     extended_explanation = extended_explanations.get(predicted_class_name, "No additional details available.")
    
#     return {
#         "original_img": img,
#         "heatmap_img": overlayed_img,
#         "predicted_class": class_name,
#         "confidence": confidence,
#         "risk_level": risk_level,
#         "explanation": extended_explanation
#     }

# # Model loading section
# st.sidebar.header("Model Configuration")
# model_path = st.sidebar.text_input(
#     "Model Path", 
#     value="models/alzheimers_effiB3_master_SE.h5",
#     help="Path to the .h5 model file"
# )

# # Load model from the specified path
# try:
#     model = tf.keras.models.load_model(model_path)
#     st.sidebar.success("Model loaded successfully!")
# except Exception as e:
#     st.sidebar.error(f"Error loading model: {e}")
#     model = None
    
# # Add option to upload a custom model if needed
# custom_model = st.sidebar.checkbox("Upload a custom model instead?")
# if custom_model:
#     model_file = st.sidebar.file_uploader("Upload model (.h5 file)", type=["h5"])
#     if model_file:
#         try:
#             # Save the model file temporarily
#             model_bytes = model_file.read()
#             with open("temp_model.h5", "wb") as f:
#                 f.write(model_bytes)
            
#             # Load the model
#             model = tf.keras.models.load_model("temp_model.h5")
#             st.sidebar.success("Custom model loaded successfully!")
#         except Exception as e:
#             st.sidebar.error(f"Error loading custom model: {e}")

# # Image upload and prediction section
# if model:
#     st.header("Upload a Retinal Image for Analysis")
#     uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])
    
#     # Add sample image option
#     # st.markdown("### Or use a sample image")
#     # use_sample = st.button("Use Sample Image")
    
#     # if use_sample:
#     #     # You would typically have this sample image in your project
#     #     # For demonstration, we'll just create a placeholder message
#     #     st.info("In a real deployment, this would load a sample retinal image from your project directory.")
#     #     # In real implementation, you'd do something like:
#     #     # uploaded_file = open("sample_images/sample_retina.jpg", "rb")
    
#     if uploaded_file is not None: #or use_sample
#         st.image(uploaded_file, caption="Uploaded Image", width=300)
#         st.write("Processing...")
        
#         try:
#             # Process the image and get results
#             results = process_image(uploaded_file, model)
            
#             # Create two columns for displaying results
#             col1, col2 = st.columns(2)
            
#             # Display original image
#             with col1:
#                 st.subheader("Original Image")
#                 st.image(results["original_img"], width=300)
            
#             # Display heatmap overlay
#             with col2:
#                 st.subheader("Grad-CAM Heatmap")
#                 st.image(results["heatmap_img"], width=300)
            
#             # Display prediction results
#             st.header("Prediction Results")
            
#             # Create a colored box for the prediction based on risk level
#             if results["risk_level"] == "low_risk":
#                 box_color = "green"
#             elif results["risk_level"] == "medium_risk":
#                 box_color = "orange"
#             else:
#                 box_color = "red"
            
#             st.markdown(
#                 f"""
#                 <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; color: white;">
#                     <h3>Predicted Condition: {results["predicted_class"]}</h3>
#                     <h4>Confidence: {results["confidence"]:.2f}%</h4>
#                     <h4>Risk Level: {results["risk_level"].replace("_", " ").title()}</h4>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
            
#             # Display explanations
#             st.header("Analysis Insights")
#             st.markdown(results["explanation"])
            
#             st.subheader("Heatmap Interpretation")
#             st.markdown(heatmap_explanation)
            
#             # Add a disclaimer
#             st.warning("""
#                 **DISCLAIMER**: This tool is for educational and research purposes only. 
#                 It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
#                 Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions.
#             """)
            
#         except Exception as e:
#             st.error(f"Error processing image: {e}")
# else:
#     st.info("Please upload the model file (.h5) first to proceed with image analysis.")

# # Add information about model path in the sidebar
# st.sidebar.markdown("---")
# st.sidebar.subheader("About")
# st.sidebar.info("""
# This application uses a pre-trained EfficientNet B3 model to predict
# Alzheimer's disease risk from retinal images. The default model path points to
# where the model should be located on your system.

# If the default path doesn't work, you can:
# 1. Change the path to match your system
# 2. Upload a custom model using the checkbox above
# """)

# # Add footer with information
# st.markdown("---")
# st.markdown("RetinaDX-NN: AI-powered Alzheimer's risk assessment through retinal analysis")
# st.markdown("Default model: EfficientNet B3 with Squeeze-and-Excitation blocks")


import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="RetinaDX-NN: Alzheimer's Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for blue theme and tabs
st.markdown("""
<style>
    :root {
        --primary-color: #1976D2;
        --secondary-color: #2196F3;
        --accent-color: #82B1FF;
        --light-bg: #E3F2FD;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: var(--primary-color);
    }
    .patient-form {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .low-risk {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .medium-risk {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
    }
    .high-risk {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .download-btn {
        width: 100%;
        margin-top: 10px;
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    .stTabs [role="tablist"] {
        gap: 10px;
    }
    .stTabs [role="tab"] {
        background-color: #E3F2FD;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 24px !important;
        border: none !important;
        color: var(--primary-color);
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    .stTabs [role="tab"]:hover {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
    .model-header {
        background-color: var(--primary-color);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .blue-button {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Model options
model_options = {
    "EfficientNet-Baseline": "models/alzheimers_effiB3_master.h5",
    "EfficientNet-CBAM": "models/alzheimers_effiB3_master_CBAM.h5",
    "EfficientNet-SE": "models/alzheimers_effiB3_master_SE.h5"
}

# Define class labels
class_labels = {
    0: ("CN (Cognitively Normal)", "low-risk"),
    1: ("MCI (Mild Cognitive Impairment)", "medium-risk"),
    2: ("AD (Alzheimer's Disease)", "high-risk")
}

# Class mapping
idx_to_class = {
    0: "CN (Cognitively Normal)",
    1: "MCI (Mild Cognitive Impairment)",
    2: "AD (Alzheimer's Disease)",
}

# Extended explanations for each class
extended_explanations = {
    "CN (Cognitively Normal)": (
        "The model did not find significant abnormalities in the retinal structure, leading to a normal classification. "
        "The absence of heatmap intensity in critical regions suggests no concerning signs of disease. "
        "A well-defined and evenly structured retina supports this assessment.\n\n"
        "Next Step: Routine eye exams are still recommended for continued eye health."
    ),
    "MCI (Mild Cognitive Impairment)": (
        "The model focused on bright, distinct deposits beneath the retina, which are characteristic of Drusen. "
        "These deposits, often found near the macula, can contribute to vision impairment if they grow larger. "
        "The heatmap highlights these abnormal deposits, reinforcing the likelihood of this condition.\n\n"
        "Next Step: Regular OCT scans are advised to monitor Drusen size and density."
    ),
    "AD (Alzheimer's Disease)": (
        "CNV (Choroidal Neovascularization)\n"
        "The model highlighted abnormal vascular regions, often associated with excessive blood vessel growth. "
        "These regions may indicate leakage or neovascularization, commonly seen in wet AMD. "
        "The heatmap shows the AI's focus on irregular patterns in the retina, which aligns with CNV characteristics.\n\n"
        "Next Step: Confirm with Fluorescein Angiography or OCT Angiography to assess neovascularization.\n\n"
        "DME (Diabetic Macular Edema)\n"
        "The model detected fluid accumulation in the macula, emphasizing regions with potential swelling. "
        "The highlighted areas suggest changes in retinal thickness, which are key indicators of macular edema. "
        "The intensity of the heatmap in the central macular zone supports this diagnosis.\n\n"
        "Next Step: Confirm with Fundus Photography or Additional OCT scans to evaluate macular thickness."
    )
}

# Heatmap color interpretation
heatmap_explanation = (
    "Red areas indicate the most critical regions influencing the AI's decision, suggesting high abnormality. "
    "Orange and yellow areas represent moderate attention, possibly indicating early signs of disease. "
    "Blue and green areas contribute the least to the decision, implying normal or less concerning regions."
)

def find_last_conv_layer(model):
    """Find the name of the last convolutional-type layer in a Keras model."""
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
    
    for layer in reversed(model.layers):
        if isinstance(layer, (Conv2D, DepthwiseConv2D)):
            return layer.name
    
    raise ValueError("No Conv2D or DepthwiseConv2D layer found in the model.")

def grad_cam(model, img_array, layer_name):
    """Apply Grad-CAM technique to visualize model's attention"""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, predicted_class.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image"""
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.uint8(255 * img_array[0])
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    
    return overlayed_img

def process_image(uploaded_file, model):
    """Process the uploaded image and return results"""
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((300, 300))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape((1, 300, 300, 3))
    
    last_conv_layer_name = find_last_conv_layer(model)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    class_name, risk_level = class_labels[predicted_class_idx]
    
    heatmap, _ = grad_cam(model, img_array, layer_name=last_conv_layer_name)
    overlayed_img = overlay_heatmap(img_array, heatmap)
    
    predicted_class_name = idx_to_class[predicted_class_idx]
    extended_explanation = extended_explanations.get(predicted_class_name, "No additional details available.")
    
    return {
        "original_img": img,
        "heatmap_img": overlayed_img,
        "predicted_class": class_name,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": extended_explanation,
        "predictions": predictions[0]
    }

def create_report(patient_info, results):
    """Generate a PDF report of the analysis"""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="RetinaDX-NN Analysis Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(10)
    
    # Patient information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Patient Information", ln=1)
    pdf.set_font("Arial", size=12)
    for key, value in patient_info.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)
    pdf.ln(10)
    
    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Analysis Results", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Condition: {results['predicted_class']}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {results['confidence']:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Risk Level: {results['risk_level'].replace('-', ' ').title()}", ln=1)
    pdf.ln(5)
    
    # Explanation
    pdf.multi_cell(0, 10, txt="Analysis Insights:")
    pdf.multi_cell(0, 10, txt=results['explanation'])
    pdf.ln(5)
    
    # Save images temporarily
    original_path = "temp_original.jpg"
    heatmap_path = "temp_heatmap.jpg"
    results['original_img'].save(original_path)
    Image.fromarray(results['heatmap_img']).save(heatmap_path)
    
    # Add images to report
    pdf.cell(200, 10, txt="Original Image vs. Heatmap Analysis", ln=1)
    pdf.image(original_path, x=10, w=90)
    pdf.image(heatmap_path, x=110, w=90)
    
    # Add disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="DISCLAIMER: This tool is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions.")
    
    # Save to bytes buffer
    report_bytes = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    report_bytes.write(pdf_bytes)
    report_bytes.seek(0)
    
    return report_bytes

# Main application
st.title("🧠 RetinaDX-NN: Alzheimer's Risk Assessment")
st.markdown("""
This professional tool analyzes retinal images to assess Alzheimer's disease risk using advanced deep learning models.
""")

# Initialize session state for model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Model selection as tabs
if not st.session_state.selected_model:
    st.markdown("### Select an AI Model for Analysis")
    
    # Create tabs for model selection
    tab1, tab2, tab3 = st.tabs(list(model_options.keys()))
    
    with tab1:
        st.markdown("""
        **EfficientNet-Baseline**  
        Standard EfficientNet B3 architecture for Alzheimer's detection from retinal images.
        """)
        if st.button("Select Baseline Model", key="baseline_btn", type="primary"):
            st.session_state.selected_model = "EfficientNet-Baseline"
            st.rerun()
    
    with tab2:
        st.markdown("""
        **EfficientNet-CBAM**  
        Enhanced with Convolutional Block Attention Module for improved feature extraction.
        """)
        if st.button("Select CBAM Model", key="cbam_btn", type="primary"):
            st.session_state.selected_model = "EfficientNet-CBAM"
            st.rerun()
    
    with tab3:
        st.markdown("""
        **EfficientNet-SE**  
        Enhanced with Squeeze-and-Excitation blocks for channel-wise attention.
        """)
        if st.button("Select SE Model", key="se_btn", type="primary"):
            st.session_state.selected_model = "EfficientNet-SE"
            st.rerun()

# If model is selected, show the analysis page
else:
    # Display selected model header
    st.markdown(f"""
    <div class="model-header">
        <h3>Selected Model: {st.session_state.selected_model}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add back button
    if st.button("← Back to Model Selection"):
        st.session_state.selected_model = None
        st.rerun()
    
    # Load the selected model
    try:
        model = tf.keras.models.load_model(model_options[st.session_state.selected_model])
        st.success(f"Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Patient information form
    with st.form("patient_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", help="Unique patient identifier")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date of Birth", min_value=datetime.date(1900, 1, 1))
            
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            email = st.text_input("Email (optional)")
            phone = st.text_input("Phone (optional)")
            exam_date = st.date_input("Examination Date", value=datetime.date.today())
        
        # Image upload section
        st.subheader("Retinal Image Upload")
        uploaded_file = st.file_uploader(
            "Upload retinal image (JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear retinal image for analysis"
        )
        
        submitted = st.form_submit_button("Analyze Image", type="primary")
    
    # Process the form submission
    if submitted and uploaded_file is not None:
        with st.spinner("Analyzing retinal image..."):
            try:
                # Process the image
                results = process_image(uploaded_file, model)
                
                # Store patient information
                patient_info = {
                    "Patient ID": patient_id,
                    "Name": f"{first_name} {last_name}",
                    "Date of Birth": dob.strftime("%Y-%m-%d"),
                    "Gender": gender,
                    "Examination Date": exam_date.strftime("%Y-%m-%d"),
                    "Contact": f"{phone} | {email}" if phone or email else "Not provided"
                }
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(results["original_img"], use_column_width=True)
                    
                with col2:
                    st.subheader("AI Heatmap Analysis")
                    st.image(results["heatmap_img"], use_column_width=True)
                
                # Display prediction results
                st.subheader("Diagnostic Results")
                
                # Determine the appropriate CSS class based on risk level
                risk_class = results["risk_level"].replace("_", "-")
                
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h3>Predicted Condition: {results["predicted_class"]}</h3>
                    <p><strong>Confidence:</strong> {results["confidence"]:.2f}%</p>
                    <p><strong>Risk Level:</strong> {results["risk_level"].replace("-", " ").title()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed explanation
                st.subheader("Clinical Interpretation")
                st.markdown(results["explanation"])
                
                st.subheader("Heatmap Interpretation")
                st.markdown(heatmap_explanation)
                
                # Generate and offer download of the report
                st.subheader("Download Report")
                report_bytes = create_report(patient_info, results)
                st.download_button(
                    label="Download Full Report (PDF)",
                    data=report_bytes,
                    file_name=f"RetinaDX_Report_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    help="Download a comprehensive PDF report of this analysis",
                    type="primary"
                )
                
                # Add disclaimer
                st.warning("""
                **DISCLAIMER**: This tool is for educational and research purposes only. 
                It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
                Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions.
                """)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    elif submitted and uploaded_file is None:
        st.error("Please upload a retinal image for analysis")

# Footer
st.markdown("---")
st.markdown("© 2023 RetinaDX-NN | AI-powered Alzheimer's risk assessment through retinal analysis")