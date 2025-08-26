import streamlit as st
from compare import compare_files
import os

# Set folder to save uploads
UPLOAD_FOLDER = "uploaded_audios"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create folder if not exists

# Streamlit page setup
st.set_page_config(page_title="Voice Matcher — Demo", page_icon="🎤", layout="wide")
st.title("🎤 Voice Matcher — Demo")
st.markdown(
    "Upload **two audio files** to compare their similarity. "
    "You can also listen to the files before comparing them."
)

# Variables for file paths
path1, path2 = None, None

# Layout with two columns for upload
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("🎵 Select first audio file", type=["wav", "mp3", "flac", "ogg"], key="file1")
    if file1:
        st.audio(file1, format="audio/wav")
        path1 = os.path.join(UPLOAD_FOLDER, file1.name)
        with open(path1, "wb") as f:
            f.write(file1.getbuffer())

with col2:
    file2 = st.file_uploader("🎵 Select second audio file", type=["wav", "mp3", "flac", "ogg"], key="file2")
    if file2:
        st.audio(file2, format="audio/wav")
        path2 = os.path.join(UPLOAD_FOLDER, file2.name)
        with open(path2, "wb") as f:
            f.write(file2.getbuffer())

st.markdown("---")

# Compare button
if st.button("Compare Voices"):
    if not path1 or not path2:
        st.warning("⚠️ Please upload **two audio files** first.")
    else:
        with st.spinner("🔄 Comparing files..."):
            try:
                # Use the saved paths for comparison
                res = compare_files(path1, path2)

                st.subheader("📊 Comparison Results")
                st.markdown(f"""
                    **Cosine Similarity:** `{res['cosine_similarity']:.3f}`  
                    **DTW Similarity:** `{res['dtw_similarity']:.3f}`  
                    **Combined Score:** `{res['combined_score']:.3f}`  
                    **Verdict:** **{res['verdict']}**
                """)
            except Exception as e:
                st.error(f"❌ Error: {e}")
