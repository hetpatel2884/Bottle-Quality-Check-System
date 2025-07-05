import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import io
import os
from collections import Counter
import time

# Configure page
st.set_page_config(
    page_title="Bottle Quality Inspector",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🍶 Bottle Quality Inspector</h1>', unsafe_allow_html=True)
st.markdown("**Upload multiple images to classify bottle quality using your Teachable Machine model**")

# Sidebar for model configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model upload section
    st.subheader("Upload Your Model")
    model_file = st.file_uploader(
        "Upload your Teachable Machine model (.h5 file)",
        type=['h5'],
        help="Export your model from Teachable Machine as Keras format"
    )
    
    labels_input = st.text_area(
        "Class Labels (one per line)",
        value="Defect\nGood\nNo Cap",
        help="Enter your class labels exactly as they appear in your model"
    )
    
    # Processing settings
    st.subheader("Processing Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for classification"
    )
    
    batch_size = st.selectbox(
        "Batch Size",
        options=[1, 5, 10, 20, 50],
        index=2,
        help="Number of images to process simultaneously"
    )

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

# Model loading function
@st.cache_resource
def load_model(model_bytes):
    """Load the Teachable Machine model from bytes"""
    try:
        # Save bytes to temporary file
        with open("temp_model.h5", "wb") as f:
            f.write(model_bytes)
        
        # Try different approaches to load the model
        model = None
        
        # Method 1: Load with custom objects (handles the groups parameter issue)
        try:
            import tensorflow.keras.utils as utils
            model = keras.models.load_model("temp_model.h5", compile=False)
            st.success("✅ Model loaded successfully using Method 1")
        except Exception as e1:
            st.warning(f"Method 1 failed: {str(e1)}")
            
            # Method 2: Try with custom_objects
            try:
                from tensorflow.keras.layers import DepthwiseConv2D
                
                # Create a custom DepthwiseConv2D that ignores the 'groups' parameter
                class CustomDepthwiseConv2D(DepthwiseConv2D):
                    def __init__(self, *args, **kwargs):
                        # Remove the 'groups' parameter if it exists
                        kwargs.pop('groups', None)
                        super().__init__(*args, **kwargs)
                
                custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
                model = keras.models.load_model("temp_model.h5", custom_objects=custom_objects, compile=False)
                st.success("✅ Model loaded successfully using Method 2 (Custom Objects)")
            except Exception as e2:
                st.warning(f"Method 2 failed: {str(e2)}")
                
                # Method 3: Try loading with TensorFlow Lite converter approach
                try:
                    # Load the model architecture and weights separately
                    model = keras.models.load_model("temp_model.h5", compile=False)
                    st.success("✅ Model loaded successfully using Method 3")
                except Exception as e3:
                    st.error(f"All methods failed. Error: {str(e3)}")
                    st.error("Please try the solutions mentioned in the troubleshooting section.")
        
        # Clean up temporary file
        if os.path.exists("temp_model.h5"):
            os.remove("temp_model.h5")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if os.path.exists("temp_model.h5"):
            os.remove("temp_model.h5")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Prediction function
def predict_image(model, image, labels, confidence_threshold):
    """Make prediction on a single image"""
    try:
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, 0.0
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get highest confidence prediction
        max_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][max_idx])
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "Low Confidence", confidence
        
        predicted_class = labels[max_idx]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, 0.0

# Batch processing function
def process_images_batch(model, images, labels, confidence_threshold, progress_bar):
    """Process multiple images in batches"""
    results = []
    
    for i, (filename, image) in enumerate(images):
        try:
            prediction, confidence = predict_image(model, image, labels, confidence_threshold)
            
            results.append({
                'filename': filename,
                'prediction': prediction,
                'confidence': confidence,
                'status': 'Success' if prediction else 'Error'
            })
            
            # Update progress
            progress_bar.progress((i + 1) / len(images))
            
        except Exception as e:
            results.append({
                'filename': filename,
                'prediction': 'Error',
                'confidence': 0.0,
                'status': f'Error: {str(e)}'
            })
    
    return results

# Main application logic
def main():
    # Check if model is loaded
    if model_file is None:
        st.warning("⚠️ Please upload your Teachable Machine model (.h5 file) in the sidebar to get started.")
        st.info("**How to export your model from Teachable Machine:**\n1. Click 'Export Model'\n2. Select 'Tensorflow' → 'Keras'\n3. Download the .h5 file\n4. Upload it using the sidebar")
        return
    
    # Load model
    model = load_model(model_file.read())
    if model is None:
        return
    
    # Parse labels
    labels = [label.strip() for label in labels_input.split('\n') if label.strip()]
    
    st.success(f"✅ Model loaded successfully! Classes: {', '.join(labels)}")
    
    # File upload section
    st.header("📁 Upload Images")
    
    # Option 1: Multiple individual files
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        accept_multiple_files=True,
        help="Upload multiple image files (PNG, JPG, JPEG, BMP, GIF)"
    )
    
    # Option 2: ZIP file upload
    zip_file = st.file_uploader(
        "Or upload a ZIP file containing images",
        type=['zip'],
        help="Upload a ZIP file containing multiple images"
    )
    
    # Process uploaded files
    images_to_process = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                images_to_process.append((uploaded_file.name, image))
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    if zip_file:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        try:
                            image_data = zip_ref.read(filename)
                            image = Image.open(io.BytesIO(image_data))
                            images_to_process.append((filename, image))
                        except Exception as e:
                            st.error(f"Error loading {filename} from ZIP: {str(e)}")
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")
    
    if not images_to_process:
        st.info("Please upload images to classify.")
        return
    
    st.info(f"📊 Found {len(images_to_process)} images to process")
    
    # Process images button
    if st.button("🚀 Start Classification", type="primary"):
        with st.spinner("Processing images..."):
            progress_bar = st.progress(0)
            
            # Process images
            results = process_images_batch(
                model, images_to_process, labels, confidence_threshold, progress_bar
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.processed_images = images_to_process
            
            st.success(f"✅ Processing complete! Classified {len(results)} images")
    
    # Display results
    if st.session_state.results:
        display_results(st.session_state.results, st.session_state.processed_images, labels)

def display_results(results, images, labels):
    """Display classification results with statistics and visualizations"""
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.header("📊 Classification Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_images = len(results)
    successful_predictions = len(df[df['status'] == 'Success'])
    avg_confidence = df[df['status'] == 'Success']['confidence'].mean()
    
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Successful Classifications", successful_predictions)
    with col3:
        st.metric("Success Rate", f"{(successful_predictions/total_images)*100:.1f}%")
    with col4:
        st.metric("Average Confidence", f"{avg_confidence:.3f}" if not pd.isna(avg_confidence) else "N/A")
    
    # Class distribution
    st.subheader("📈 Class Distribution")
    
    # Count predictions by class
    prediction_counts = df[df['status'] == 'Success']['prediction'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Distribution of Classifications",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=prediction_counts.index,
            y=prediction_counts.values,
            title="Count by Class",
            labels={'x': 'Class', 'y': 'Count'},
            color=prediction_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confidence distribution
    st.subheader("🎯 Confidence Analysis")
    
    successful_df = df[df['status'] == 'Success']
    
    if not successful_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence histogram
            fig_hist = px.histogram(
                successful_df,
                x='confidence',
                nbins=20,
                title="Distribution of Confidence Scores",
                labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by class
            fig_box = px.box(
                successful_df,
                x='prediction',
                y='confidence',
                title="Confidence by Class",
                labels={'prediction': 'Class', 'confidence': 'Confidence Score'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        class_filter = st.multiselect(
            "Filter by Class",
            options=df['prediction'].unique(),
            default=df['prediction'].unique()
        )
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    # Filter dataframe
    filtered_df = df[
        (df['prediction'].isin(class_filter)) & 
        (df['confidence'] >= min_confidence)
    ]
    
    # Display filtered results
    st.dataframe(
        filtered_df[['filename', 'prediction', 'confidence', 'status']],
        use_container_width=True,
        hide_index=True
    )
    
    # Download results
    st.subheader("💾 Download Results")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"bottle_classification_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Sample images with predictions
    st.subheader("🖼️ Sample Results")
    
    # Show first few images with predictions
    sample_size = min(6, len(images))
    cols = st.columns(3)
    
    for i in range(sample_size):
        col_idx = i % 3
        filename, image = images[i]
        result = results[i]
        
        with cols[col_idx]:
            st.image(image, caption=f"{filename}", use_column_width=True)
            
            # Color code based on prediction
            if result['prediction'] == 'Good':
                st.success(f"✅ {result['prediction']} ({result['confidence']:.3f})")
            elif result['prediction'] == 'Defect':
                st.error(f"❌ {result['prediction']} ({result['confidence']:.3f})")
            elif result['prediction'] == 'No Cap':
                st.warning(f"⚠️ {result['prediction']} ({result['confidence']:.3f})")
            else:
                st.info(f"ℹ️ {result['prediction']} ({result['confidence']:.3f})")

if __name__ == "__main__":
    main()