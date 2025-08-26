import numpy as np
import librosa
from scipy.spatial.distance import cosine
import librosa.sequence
import io

def load_audio(source, sr=16000, mono=True):
    """
    Load audio from a file path or a file-like object (BytesIO)
    """
    if isinstance(source, (str, bytes)):
        # string path
        y, _ = librosa.load(source, sr=sr, mono=mono)
        name = source if isinstance(source, str) else "BytesIO"
    elif isinstance(source, io.BytesIO):
        source.seek(0)  # make sure we're at the start
        y, _ = librosa.load(source, sr=sr, mono=mono)
        name = "BytesIO"
    else:
        raise TypeError("source must be a file path or BytesIO object")

    if y.size == 0:
        raise ValueError(f"Loaded audio is empty: {name}")
    print(f"[INFO] Loaded '{name}' with {len(y)} samples at {sr} Hz")
    return y, sr

def extract_mfcc(y, sr, n_mfcc=13, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    print(f"[INFO] Extracted MFCCs: shape = {mfcc.shape}")
    return mfcc

def embedding_from_mfcc(mfcc):
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    emb = np.concatenate([mean, std])
    print(f"[INFO] Embedding vector size = {emb.shape}")
    return emb

def cosine_similarity(a, b):
    d = cosine(a, b)
    if np.isnan(d):
        return 0.0
    return 1.0 - d

def normalized_dtw_distance(m1, m2):
    D, wp = librosa.sequence.dtw(X=m1, Y=m2, metric='euclidean')
    cost = D[-1, -1]
    path_length = len(wp)
    if path_length == 0:
        return 1.0
    norm_cost = cost / path_length
    sim = np.exp(-norm_cost / 50.0)
    return float(sim)

def compare_files(path1, path2, in_memory=False):
    """
    path1, path2: either file paths or BytesIO objects
    in_memory: if True, treat as BytesIO
    """
    y1, sr1 = load_audio(path1 if not in_memory else path1)
    y2, sr2 = load_audio(path2 if not in_memory else path2)

    mfcc1 = extract_mfcc(y1, sr1)
    mfcc2 = extract_mfcc(y2, sr2)

    emb1 = embedding_from_mfcc(mfcc1)
    emb2 = embedding_from_mfcc(mfcc2)

    cos_sim = cosine_similarity(emb1, emb2)
    dtw_sim = normalized_dtw_distance(mfcc1, mfcc2)

    combined = 0.6 * cos_sim + 0.4 * dtw_sim
    verdict = "SAME PERSON" if combined >= 0.65 else "DIFFERENT PERSON"

    print(f"[RESULT] Cosine similarity: {cos_sim:.4f}")
    print(f"[RESULT] DTW similarity: {dtw_sim:.4f}")
    print(f"[RESULT] Combined score: {combined:.4f} -> {verdict}")

    return {
        "cosine_similarity": float(cos_sim),
        "dtw_similarity": float(dtw_sim),
        "combined_score": float(combined),
        "verdict": verdict
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python compare.py file1.wav file2.wav")
        sys.exit(1)
    out = compare_files(sys.argv[1], sys.argv[2])
    print(out)
