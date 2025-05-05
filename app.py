import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="RetinaDX-NN: Alzheimer's Prediction",
    page_icon="🧠",
    layout="wide"
)

# Add title and description
st.title("RetinaDX-NN: Alzheimer's Prediction from Retinal Images")
st.markdown("""
This application uses a deep learning model to predict Alzheimer's disease risk based on retinal images.
Upload a retinal image to get a prediction and visualization of the model's attention.
""")

# Define class labels
class_labels = {
    0: ("CN (Cognitively Normal)", "low_risk"),
    1: ("MCI (Mild Cognitive Impairment)", "medium_risk"),
    2: ("AD (Alzheimer's Disease)", "high_risk")
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
        "\nThe model did not find significant abnormalities in the retinal structure, leading to a normal classification. "
        "\nThe absence of heatmap intensity in critical regions suggests no concerning signs of disease. "
        "\nA well-defined and evenly structured retina supports this assessment."
        "\n\nNext Step: Routine eye exams are still recommended for continued eye health."
    ),
    "MCI (Mild Cognitive Impairment)": (
        "\nThe model focused on bright, distinct deposits beneath the retina, which are characteristic of Drusen. "
        "\nThese deposits, often found near the macula, can contribute to vision impairment if they grow larger. "
        "\nThe heatmap highlights these abnormal deposits, reinforcing the likelihood of this condition."
        "\n\nNext Step: Regular OCT scans are advised to monitor Drusen size and density."
    ),
    "AD (Alzheimer's Disease)": (
        "\nCNV (Choroidal Neovascularization)"
        "\nThe model highlighted abnormal vascular regions, often associated with excessive blood vessel growth. "
        "\nThese regions may indicate leakage or neovascularization, commonly seen in wet AMD. "
        "\nThe heatmap shows the AI's focus on irregular patterns in the retina, which aligns with CNV characteristics."
        "\n\nNext Step: Confirm with Fluorescein Angiography or OCT Angiography to assess neovascularization."
        "\n\nDME (Diabetic Macular Edema)"
        "\nThe model detected fluid accumulation in the macula, emphasizing regions with potential swelling. "
        "\nThe highlighted areas suggest changes in retinal thickness, which are key indicators of macular edema. "
        "\nThe intensity of the heatmap in the central macular zone supports this diagnosis."
        "\n\nNext Step: Confirm with Fundus Photography or Additional OCT scans to evaluate macular thickness."
    )
}

# Heatmap color interpretation
heatmap_explanation = (
    "\nRed areas indicate the most critical regions influencing the AI's decision, suggesting high abnormality. "
    "\nOrange and yellow areas represent moderate attention, possibly indicating early signs of disease. "
    "\nBlue and green areas contribute the least to the decision, implying normal or less concerning regions."
)

def find_last_conv_layer(model):
    """
    Find the name of the last convolutional-type layer (Conv2D or DepthwiseConv2D) in a Keras model.
    """
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
    
    # Iterate layers in reverse so we find the last one first
    for layer in reversed(model.layers):
        # Check if this layer is a standard or depthwise convolution
        if isinstance(layer, (Conv2D, DepthwiseConv2D)):
            return layer.name
    
    # If loop completes without returning, no conv-type layer was found
    raise ValueError("No Conv2D or DepthwiseConv2D layer found in the model.")

def grad_cam(model, img_array, layer_name):
    """Apply Grad-CAM technique to visualize model's attention"""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])  # Get the predicted class index
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)  # Compute gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)  # Weighted activation

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap, predicted_class.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image"""
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))  # Resize heatmap
    heatmap = np.uint8(255 * heatmap)  # Convert to RGB format
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

    img = np.uint8(255 * img_array[0])  # Convert image to correct format
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Merge heatmap with image
    
    # Convert from BGR (OpenCV default) to RGB for matplotlib/streamlit
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    
    return overlayed_img

def process_image(uploaded_file, model):
    # Open and preprocess image
    img = Image.open(uploaded_file).convert('RGB')
    # Resize to the expected input shape of the model (300, 300) and keep RGB channels
    img = img.resize((300, 300))
    img_array = np.array(img).astype('float32') / 255.0
    # Add batch dimension
    img_array = img_array.reshape((1, 300, 300, 3))
    
    # Get the last convolutional layer
    last_conv_layer_name = find_last_conv_layer(model)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get class name and risk level
    class_name, risk_level = class_labels[predicted_class_idx]
    
    # Generate Grad-CAM visualization
    heatmap, _ = grad_cam(model, img_array, layer_name=last_conv_layer_name)
    overlayed_img = overlay_heatmap(img_array, heatmap)
    
    # Get extended explanation
    predicted_class_name = idx_to_class[predicted_class_idx]
    extended_explanation = extended_explanations.get(predicted_class_name, "No additional details available.")
    
    return {
        "original_img": img,
        "heatmap_img": overlayed_img,
        "predicted_class": class_name,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": extended_explanation
    }

# Model loading section
st.sidebar.header("Model Configuration")
model_path = st.sidebar.text_input(
    "Model Path", 
    value="models/alzheimers_effiB3_master_SE.h5",
    help="Path to the .h5 model file"
)

# Load model from the specified path
try:
    model = tf.keras.models.load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None
    
# Add option to upload a custom model if needed
custom_model = st.sidebar.checkbox("Upload a custom model instead?")
if custom_model:
    model_file = st.sidebar.file_uploader("Upload model (.h5 file)", type=["h5"])
    if model_file:
        try:
            # Save the model file temporarily
            model_bytes = model_file.read()
            with open("temp_model.h5", "wb") as f:
                f.write(model_bytes)
            
            # Load the model
            model = tf.keras.models.load_model("temp_model.h5")
            st.sidebar.success("Custom model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading custom model: {e}")

# Image upload and prediction section
if model:
    st.header("Upload a Retinal Image for Analysis")
    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])
    
    # Add sample image option
    # st.markdown("### Or use a sample image")
    # use_sample = st.button("Use Sample Image")
    
    # if use_sample:
    #     # You would typically have this sample image in your project
    #     # For demonstration, we'll just create a placeholder message
    #     st.info("In a real deployment, this would load a sample retinal image from your project directory.")
    #     # In real implementation, you'd do something like:
    #     # uploaded_file = open("sample_images/sample_retina.jpg", "rb")
    
    if uploaded_file is not None: #or use_sample
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.write("Processing...")
        
        try:
            # Process the image and get results
            results = process_image(uploaded_file, model)
            
            # Create two columns for displaying results
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                st.image(results["original_img"], width=300)
            
            # Display heatmap overlay
            with col2:
                st.subheader("Grad-CAM Heatmap")
                st.image(results["heatmap_img"], width=300)
            
            # Display prediction results
            st.header("Prediction Results")
            
            # Create a colored box for the prediction based on risk level
            if results["risk_level"] == "low_risk":
                box_color = "green"
            elif results["risk_level"] == "medium_risk":
                box_color = "orange"
            else:
                box_color = "red"
            
            st.markdown(
                f"""
                <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; color: white;">
                    <h3>Predicted Condition: {results["predicted_class"]}</h3>
                    <h4>Confidence: {results["confidence"]:.2f}%</h4>
                    <h4>Risk Level: {results["risk_level"].replace("_", " ").title()}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display explanations
            st.header("Analysis Insights")
            st.markdown(results["explanation"])
            
            st.subheader("Heatmap Interpretation")
            st.markdown(heatmap_explanation)
            
            # Add a disclaimer
            st.warning("""
                **DISCLAIMER**: This tool is for educational and research purposes only. 
                It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
                Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions.
            """)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.info("Please upload the model file (.h5) first to proceed with image analysis.")

# Add information about model path in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This application uses a pre-trained EfficientNet B3 model to predict
Alzheimer's disease risk from retinal images. The default model path points to
where the model should be located on your system.

If the default path doesn't work, you can:
1. Change the path to match your system
2. Upload a custom model using the checkbox above
""")

# Add footer with information
st.markdown("---")
st.markdown("RetinaDX-NN: AI-powered Alzheimer's risk assessment through retinal analysis")
st.markdown("Default model: EfficientNet B3 with Squeeze-and-Excitation blocks")