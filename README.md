# Voice Matcher — Demo Python App
This demo compares two voice files and estimates whether they are from the same speaker.
It is a simple, educational project — not production-grade speaker verification.

## What's inside
- `main.py` — Tkinter GUI to choose two audio files and run comparison.
- `compare.py` — Core comparison: loads audio, computes MFCCs, compares by cosine similarity
                and DTW, and returns a combined score.
- `requirements.txt` — Python packages you should install.
- `run_demo.sh` — (Optional) Linux / WSL helper to create virtualenv and run the app.

## How it works (short)
1. Load both audio files (WAV/MP3) using `librosa`.
2. Extract MFCC features (13 coefficients).
3. Create a single embedding per file by averaging MFCC frames.
4. Compute cosine similarity between embeddings and normalized DTW distance between MFCC sequences.
5. Combine both measures into a single "similarity score" (0..1). If score >= 0.65 => "SAME PERSON" (tunable).

## Quick start
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # linux / mac
   venv\Scripts\activate    # windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the GUI:
   ```bash
   python main.py
   ```

## Notes and limitations
- This demo is for learning and prototypes. Real speaker verification uses trained embedding models (e.g. d-vector, x-vector, ECAPA-TDNN).
- Results depend on audio quality, length, noise, and channel mismatch.
- You can adjust the threshold in `compare.py` or replace the embedding method with a pretrained model later.

## Contact
If you want me to add a pretrained embedding model (better accuracy) or pack this into an executable, tell me and I’ll update the project.\n"# voicerecognition" 
