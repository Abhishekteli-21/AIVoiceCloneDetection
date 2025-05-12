

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
# import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from io import BytesIO

# Import custom modules (assumed to exist)
import config
from feature_extraction import extract_mfcc_features, create_dataset
from model_training import train_model
from audio_classifier import analyze_audio

# Set page configuration
st.set_page_config(
    page_title="DeepFake Audio Detector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories
os.makedirs('data/real', exist_ok=True)
os.makedirs('data/fake', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Check if model exists
def model_exists():
    model_path = os.path.join('models', config.MODEL_FILENAME)
    scaler_path = os.path.join('models', config.SCALER_FILENAME)
    return os.path.exists(model_path) and os.path.exists(scaler_path)

# Function to pad or handle short audio
def load_audio_safe(audio_path, min_length=2048):
    y, sr = librosa.load(audio_path)
    if len(y) == 0:
        # Return a small silent signal if the audio is empty
        y = np.zeros(min_length)
    elif len(y) < min_length:
        # Pad with zeros if too short
        y = librosa.util.pad_center(y, size=min_length)
    return y, sr

# Modified waveform plotting function
def plot_waveform(audio_path):
    try:
        y, sr = load_audio_safe(audio_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Audio Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return fig
    except Exception as e:
        st.warning(f"Could not plot waveform: {e}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Waveform not available (audio too short)", ha='center', va='center')
        ax.set_axis_off()
        return fig

# Modified MFCC plotting function
def plot_mfcc(audio_path):
    try:
        y, sr = load_audio_safe(audio_path)
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=config.MFCC_PARAMS['n_mfcc'],
            n_fft=min(2048, len(y)),  # Adjust n_fft dynamically
            hop_length=config.MFCC_PARAMS['hop_length']
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mfccs,
            x_axis='time',
            ax=ax,
            sr=sr,
            hop_length=config.MFCC_PARAMS['hop_length']
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('MFCC Features')
        return fig
    except Exception as e:
        st.warning(f"Could not plot MFCC: {e}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "MFCC not available (audio too short)", ha='center', va='center')
        ax.set_axis_off()
        return fig

# Function to save uploaded file
def save_uploaded_file(uploaded_file, directory):
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Sidebar navigation (unchanged)
st.sidebar.markdown("<h1 style='text-align: center;'>Navigation</h1>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Train Model", "Analyze Audio", "Batch Analysis", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center;'>About</h3>", unsafe_allow_html=True)
st.sidebar.info("""
This application uses machine learning to detect deepfake audio files.
It extracts MFCC features from audio samples and uses an SVM classifier
to distinguish between genuine and fake audio.
""")

# Home page (unchanged)
if page == "Home":
    st.markdown("<h1 class='main-header'>DeepFake Audio Detection System</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("image.png", use_container_width=True)
    
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    ## What are Deepfake Audios?
    
    Deepfake audio refers to synthetic audio content created using artificial intelligence to mimic a person's voice. 
    These technologies can create convincing imitations that sound like specific individuals saying things they never actually said.
    
    ## How This Application Works
    
    This application uses machine learning to detect whether an audio file is genuine or a deepfake:
    
    1. **Feature Extraction**: The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files, which capture the timbral and spectral characteristics of the audio.
    
    2. **Model Training**: An SVM (Support Vector Machine) classifier is trained on a dataset of real and fake audio samples.
    
    3. **Classification**: When you upload an audio file, the system extracts its features and uses the trained model to classify it as genuine or deepfake.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. **Train Model**: Upload real and fake audio samples to train the model.
    2. **Analyze Audio**: Upload an audio file to determine if it's genuine or a deepfake.
    3. **Batch Analysis**: Analyze multiple audio files at once.
    4. **Model Performance**: View the performance metrics of the trained model.
    """)
    
    if model_exists():
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("✅ **Model Status**: A trained model is available and ready to use.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.markdown("⚠️ **Model Status**: No trained model found. Please go to the 'Train Model' page to train a model.")
        st.markdown("</div>", unsafe_allow_html=True)

# Train Model page (unchanged)
elif page == "Train Model":
    st.markdown("<h1 class='main-header'>Train Deepfake Detection Model</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    Upload audio samples to train the model. You need to provide:
    - **Real audio samples**: Genuine audio recordings
    - **Fake audio samples**: Deepfake or synthetic audio recordings
    
    The system will extract MFCC features from these samples and train an SVM classifier.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Upload Real Audio Samples</h3>", unsafe_allow_html=True)
        real_audio_files = st.file_uploader("Upload real audio files (.wav)", type=["wav"], accept_multiple_files=True, key="real_audio")
        
        if real_audio_files:
            st.success(f"{len(real_audio_files)} real audio files uploaded")
            if st.button("Save Real Audio Files"):
                progress_bar = st.progress(0)
                for i, file in enumerate(real_audio_files):
                    save_uploaded_file(file, config.REAL_AUDIO_DIR)
                    progress_bar.progress((i + 1) / len(real_audio_files))
                st.success("Real audio files saved successfully!")
    
    with col2:
        st.markdown("<h3 class='sub-header'>Upload Fake Audio Samples</h3>", unsafe_allow_html=True)
        fake_audio_files = st.file_uploader("Upload fake audio files (.wav)", type=["wav"], accept_multiple_files=True, key="fake_audio")
        
        if fake_audio_files:
            st.success(f"{len(fake_audio_files)} fake audio files uploaded")
            if st.button("Save Fake Audio Files"):
                progress_bar = st.progress(0)
                for i, file in enumerate(fake_audio_files):
                    save_uploaded_file(file, config.FAKE_AUDIO_DIR)
                    progress_bar.progress((i + 1) / len(fake_audio_files))
                st.success("Fake audio files saved successfully!")
    
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>Dataset Statistics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        real_files = [f for f in os.listdir(config.REAL_AUDIO_DIR) if f.endswith('.wav')]
        st.metric("Real Audio Samples", len(real_files))
    with col2:
        fake_files = [f for f in os.listdir(config.FAKE_AUDIO_DIR) if f.endswith('.wav')]
        st.metric("Fake Audio Samples", len(fake_files))
    
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>Train Model</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        train_button = st.button("Train Model", use_container_width=True)
    with col2:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=config.TRAIN_CONFIG['test_size'], step=0.05)
    
    if train_button:
        if len(real_files) < 2 or len(fake_files) < 2:
            st.error("Each class should have at least two samples. Please upload more audio files.")
        else:
            with st.spinner("Training model... This may take a while."):
                X_real, y_real = create_dataset(config.REAL_AUDIO_DIR, label=0)
                X_fake, y_fake = create_dataset(config.FAKE_AUDIO_DIR, label=1)
                X = np.vstack((X_real, X_fake))
                y = np.hstack((y_real, y_fake))
                config.TRAIN_CONFIG['test_size'] = test_size
                try:
                    train_model(X, y)
                    st.success("Model trained successfully!")
                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    **Model Information**:
                    - Type: Support Vector Machine (SVM)
                    - Kernel: Linear
                    - Features: {config.MFCC_PARAMS['n_mfcc']} MFCC coefficients
                    - Training samples: {len(X)}
                    - Test size: {test_size}
                    """)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error training model: {e}")

# Analyze Audio page (modified)
elif page == "Analyze Audio":
    st.markdown("<h1 class='main-header'>Analyze Audio</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    Upload an audio file to analyze whether it's genuine or a deepfake.
    The system will extract MFCC features from the audio and use the trained model to classify it.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not model_exists():
        st.markdown("<div class='error-box'>", unsafe_allow_html=True)
        st.markdown("⚠️ **No trained model found**. Please go to the 'Train Model' page to train a model first.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
        
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file, "data/test")
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio..."):
                    result = analyze_audio(file_path)
                    if result == "genuine":
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.markdown("## Result: ✅ Genuine Audio")
                        st.markdown("The audio file detected to be genuine and not manipulated.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif result == "deepfake":
                        st.markdown("<div class='error-box'>", unsafe_allow_html=True)
                        st.markdown("## Result: ⚠️ Deepfake Detected")
                        st.markdown("The audio detected to be synthetic or manipulated.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Analysis error: {result}")
                    
                    st.markdown("---")
                    st.markdown("<h3 class='sub-header'>Audio Visualizations</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Waveform")
                        waveform_fig = plot_waveform(file_path)
                        st.pyplot(waveform_fig)
                    with col2:
                        st.markdown("### MFCC Features")
                        mfcc_fig = plot_mfcc(file_path)
                        st.pyplot(mfcc_fig)
                    
                    st.markdown("---")
                    st.markdown("<h3 class='sub-header'>Feature Analysis</h3>", unsafe_allow_html=True)
                    try:
                        mfcc_features = extract_mfcc_features(file_path, **config.MFCC_PARAMS)
                        fig = px.bar(
                            x=[f"MFCC {i+1}" for i in range(len(mfcc_features))],
                            y=mfcc_features,
                            title="MFCC Coefficients",
                            labels={"x": "Coefficient", "y": "Value"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not extract MFCC features: {e}")

# Batch Analysis page (unchanged)
elif page == "Batch Analysis":
    st.markdown("<h1 class='main-header'>Batch Audio Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    Upload multiple audio files for batch analysis.
    The system will process each file and provide classification results.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not model_exists():
        st.markdown("<div class='error-box'>", unsafe_allow_html=True)
        st.markdown("⚠️ **No trained model found**. Please go to the 'Train Model' page to train a model first.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        uploaded_files = st.file_uploader("Upload audio files (.wav)", type=["wav"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded")
            if st.button("Analyze All Files"):
                results_df = pd.DataFrame(columns=["Filename", "Classification", "Confidence"])
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    file_path = save_uploaded_file(file, "data/test")
                    result = analyze_audio(file_path)
                    confidence = 0.95 if result == "genuine" else 0.92  # Placeholder values
                    results_df = pd.concat([results_df, pd.DataFrame({
                        "Filename": [file.name],
                        "Classification": [result],
                        "Confidence": [confidence]
                    })], ignore_index=True)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Analysis complete!")
                st.markdown("---")
                st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
                
                def highlight_classification(val):
                    if val == "genuine":
                        return "background-color: #E8F5E9; color: #2E7D32"
                    elif val == "deepfake":
                        return "background-color: #FFEBEE; color: #C62828"
                    return ""
                
                styled_df = results_df.style.applymap(highlight_classification, subset=["Classification"])
                st.dataframe(styled_df, use_container_width=True)
                
                genuine_count = (results_df["Classification"] == "genuine").sum()
                fake_count = (results_df["Classification"] == "deepfake").sum()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Genuine Audio Files", genuine_count)
                with col2:
                    st.metric("Deepfake Audio Files", fake_count)
                
                fig = px.pie(
                    names=["Genuine", "Deepfake"],
                    values=[genuine_count, fake_count],
                    title="Classification Results",
                    color_discrete_sequence=["#4CAF50", "#F44336"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="deepfake_analysis_results.csv",
                    mime="text/csv"
                )

# Model Performance page (unchanged)
elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
    if not model_exists():
        st.markdown("<div class='error-box'>", unsafe_allow_html=True)
        st.markdown("⚠️ **No trained model found**. Please go to the 'Train Model' page to train a model first.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        This page shows the performance metrics of the trained model.
        You can evaluate the model on a test dataset to see how well it performs.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    model_path = os.path.join('models', config.MODEL_FILENAME)
                    scaler_path = os.path.join('models', config.SCALER_FILENAME)
                    svm_classifier = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    X_real, y_real = create_dataset(config.REAL_AUDIO_DIR, label=0)
                    X_fake, y_fake = create_dataset(config.FAKE_AUDIO_DIR, label=1)
                    X = np.vstack((X_real, X_fake))
                    y = np.hstack((y_real, y_fake))
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(
                        X, y,
                        test_size=config.TRAIN_CONFIG['test_size'],
                        random_state=config.TRAIN_CONFIG['random_state'],
                        stratify=y
                    )
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = svm_classifier.predict(X_test_scaled)
                    accuracy = np.mean(y_pred == y_test)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    
                    st.markdown("---")
                    st.markdown("<h3 class='sub-header'>Performance Metrics</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Precision (Fake)", f"{class_report['1']['precision']:.2%}")
                    with col3:
                        st.metric("Recall (Fake)", f"{class_report['1']['recall']:.2%}")
                    
                    st.markdown("---")
                    st.markdown("<h3 class='sub-header'>Confusion Matrix</h3>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        conf_matrix,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Genuine", "Deepfake"],
                        yticklabels=["Genuine", "Deepfake"],
                        ax=ax
                    )
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.markdown("<h3 class='sub-header'>Classification Report</h3>", unsafe_allow_html=True)
                    report_df = pd.DataFrame(class_report).transpose()
                    report_df = report_df.drop('support', axis=1)
                    report_df = report_df.drop('accuracy', axis=0)
                    report_df.index = ["Genuine", "Deepfake", "Macro Avg", "Weighted Avg"]
                    report_df = report_df.applymap(lambda x: f"{x:.2%}")
                    st.dataframe(report_df, use_container_width=True)
                    
                    if hasattr(svm_classifier, "decision_function"):
                        from sklearn.metrics import roc_curve, auc
                        y_scores = svm_classifier.decision_function(X_test_scaled)
                        fpr, tpr, _ = roc_curve(y_test, y_scores)
                        roc_auc = auc(fpr, tpr)
                        st.markdown("---")
                        st.markdown("<h3 class='sub-header'>ROC Curve</h3>", unsafe_allow_html=True)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.2f})',
                            line=dict(color='darkorange', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(color='navy', width=2, dash='dash')
                        ))
                        fig.update_layout(
                            title='Receiver Operating Characteristic (ROC) Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            legend=dict(x=0.7, y=0.1),
                            width=700,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")