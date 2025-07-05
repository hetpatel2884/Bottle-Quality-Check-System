import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import zipfile
import io
import time
import h5py

st.set_page_config(
    page_title="Bottle Quality Inspector",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .pagination-controls {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    .pagination-btn {
        margin: 0 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        background-color: #1f77b4;
        color: white;
        cursor: pointer;
    }
    .pagination-btn:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍶 Bottle Quality Inspector</h1>', unsafe_allow_html=True)
st.markdown("**Upload multiple images to classify bottle quality using your Teachable Machine model**")

with st.sidebar:
    st.header("⚙️ Model Configuration")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    available_models = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    available_labels = [f for f in os.listdir(model_dir) if f.endswith("labels.txt")]

    model_file = st.file_uploader("Upload your model (.h5 file)", type=['h5'])
    model_bytes = None
    
    # Handle model selection/upload
    if model_file:
        model_bytes = model_file.read()
    elif available_models:
        selected_model_name = st.selectbox("Or select a model from directory", available_models)
        with open(os.path.join(model_dir, selected_model_name), "rb") as f:
            model_bytes = f.read()
    
    # Handle labels selection/input
    labels_input = "Defect\nGood\nNo Cap"
    if available_labels:
        selected_label_file = st.selectbox("Select label file", available_labels)
        with open(os.path.join(model_dir, selected_label_file), "r") as f:
            labels_input = f.read()
    labels_input = st.text_area("Class Labels (one per line)", value=labels_input)

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    batch_size = st.selectbox("Batch Size", [1, 5, 10, 20, 50], index=2)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Fixed model loading function with DepthwiseConv2D patch
@st.cache_resource
def load_model(model_bytes):
    """Safely load Teachable Machine model with DepthwiseConv2D patch"""
    temp_path = "temp_model.h5"
    
    try:
        # Save to temporary file
        with open(temp_path, "wb") as f:
            f.write(model_bytes)
            
        # Patch the model config to remove 'groups' parameter
        with h5py.File(temp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                # Remove 'groups' from DepthwiseConv2D layers
                model_config = model_config.replace('"groups": 1,', '').replace('"groups":1,', '')
                f.attrs['model_config'] = model_config.encode('utf-8')
        
        # Define custom DepthwiseConv2D class
        class PatchedDepthwiseConv2D(keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)  # Remove problematic argument
                super().__init__(*args, **kwargs)
        
        # Load model with custom layer
        model = keras.models.load_model(
            temp_path,
            compile=False,
            custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D}
        )
        
        st.success("✅ Model loaded successfully with DepthwiseConv2D patch")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Prediction function
def predict_image(model, image, labels, confidence_threshold):
    try:
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, 0.0
        
        predictions = model.predict(processed_img, verbose=0)
        max_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][max_idx])
        
        if confidence < confidence_threshold:
            return "Low Confidence", confidence
        
        return labels[max_idx], confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, 0.0

# Batch processing function
def process_images_batch(model, images, labels, confidence_threshold, progress_bar):
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
    if not model_bytes:
        st.warning("⚠️ Please upload or select a Teachable Machine model (.h5 file)")
        st.info("**How to export your model:**\n1. Click 'Export Model'\n2. Select 'Tensorflow' → 'Keras'\n3. Download the .h5 file")
        return
    
    # Load model
    model = load_model(model_bytes)
    if model is None:
        return
    
    # Parse labels
    labels = [label.strip() for label in labels_input.split('\n') if label.strip()]
    st.success(f"✅ Model loaded successfully! Classes: {', '.join(labels)}")
    
    # File upload section
    st.header("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image files", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        accept_multiple_files=True, help="Upload multiple image files"
    )
    
    zip_file = st.file_uploader(
        "Or upload a ZIP file containing images",
        type=['zip'], help="Upload ZIP with multiple images"
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
            results = process_images_batch(
                model, images_to_process, labels, confidence_threshold, progress_bar
            )
            st.session_state.results = results
            st.session_state.processed_images = images_to_process
            st.session_state.current_page = 0  # Reset to first page
            st.success(f"✅ Processing complete! Classified {len(results)} images")
    
    # Display results
    if st.session_state.results:
        display_results(st.session_state.results, st.session_state.processed_images, labels)

def display_results(results, images, labels):
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.header("📊 Classification Results")
    col1, col2, col3, col4 = st.columns(4)
    
    total_images = len(results)
    successful_predictions = len(df[df['status'] == 'Success'])
    avg_confidence = df[df['status'] == 'Success']['confidence'].mean()
    
    with col1: st.metric("Total Images", total_images)
    with col2: st.metric("Successful Classifications", successful_predictions)
    with col3: st.metric("Success Rate", f"{(successful_predictions/total_images)*100:.1f}%")
    with col4: st.metric("Average Confidence", f"{avg_confidence:.3f}" if not pd.isna(avg_confidence) else "N/A")
    
    # Class distribution
    st.subheader("📈 Class Distribution")
    prediction_counts = df[df['status'] == 'Success']['prediction'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                         title="Distribution of Classifications", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=prediction_counts.index, y=prediction_counts.values, title="Count by Class",
                         labels={'x': 'Class', 'y': 'Count'}, color=prediction_counts.index,
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confidence distribution
    st.subheader("🎯 Confidence Analysis")
    successful_df = df[df['status'] == 'Success']
    
    if not successful_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(successful_df, x='confidence', nbins=20,
                                   title="Distribution of Confidence Scores",
                                   labels={'confidence': 'Confidence Score', 'count': 'Frequency'})
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(successful_df, x='prediction', y='confidence',
                            title="Confidence by Class",
                            labels={'prediction': 'Class', 'confidence': 'Confidence Score'})
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    col1, col2 = st.columns(2)
    with col1:
        class_filter = st.multiselect("Filter by Class", options=df['prediction'].unique(),
                                     default=df['prediction'].unique())
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)
    
    filtered_df = df[(df['prediction'].isin(class_filter)) & (df['confidence'] >= min_confidence)]
    st.dataframe(filtered_df[['filename', 'prediction', 'confidence', 'status']],
                use_container_width=True, hide_index=True)
    
    # Download results
    st.subheader("💾 Download Results")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"bottle_classification_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # All images with predictions - PAGINATED VERSION
    st.subheader("🖼️ All Results")
    
    # Pagination settings
    images_per_page = 9
    total_pages = max(1, (len(images) + images_per_page - 1) // images_per_page)
    
    # Pagination controls
    st.markdown('<div class="pagination-controls">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⏮️ Previous", disabled=st.session_state.current_page == 0, 
                    key="prev_btn", use_container_width=True):
            st.session_state.current_page -= 1
            st.rerun()
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 0.5rem;'>Page {st.session_state.current_page + 1}/{total_pages}</div>", 
                   unsafe_allow_html=True)
    with col3:
        if st.button("Next ⏭️", disabled=st.session_state.current_page >= total_pages - 1, 
                    key="next_btn", use_container_width=True):
            st.session_state.current_page += 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display current page
    start_idx = st.session_state.current_page * images_per_page
    end_idx = min(start_idx + images_per_page, len(images))
    
    # Create 3 columns for image display
    cols = st.columns(3)
    
    for i in range(start_idx, end_idx):
        filename, image = images[i]
        result = results[i]
        col_idx = (i - start_idx) % 3
        
        with cols[col_idx]:
            # Display image - FIXED DEPRECATION WARNING HERE
            st.image(image, caption=filename, use_container_width=True)  # Changed to use_container_width
            
            # Display prediction with color coding
            if result['prediction'] == 'Good':
                st.success(f"✅ {result['prediction']} ({result['confidence']:.3f})")
            elif result['prediction'] == 'Defect':
                st.error(f"❌ {result['prediction']} ({result['confidence']:.3f})")
            elif result['prediction'] == 'No Cap':
                st.warning(f"⚠️ {result['prediction']} ({result['confidence']:.3f})")
            else:
                st.info(f"ℹ️ {result['prediction']} ({result['confidence']:.3f})")
            
            # Confidence progress bar
            st.progress(result['confidence'], text=f"Confidence: {result['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()