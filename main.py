import tkinter as tk
from tkinter import filedialog, messagebox
from compare import compare_files
import threading

def select_file(entry):
    path = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
    )
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def run_compare(path1, path2, result_text, compare_button):
    try:
        compare_button.config(state=tk.DISABLED)
        result_text.set("Processing... (this may take a few seconds)")
        res = compare_files(path1, path2)
        txt = (
            f"Cosine similarity: {res['cosine_similarity']:.3f}\n"
            f"DTW similarity: {res['dtw_similarity']:.3f}\n"
            f"Combined score: {res['combined_score']:.3f}\n\n"
            f"Verdict: {res['verdict']}"
        )
        result_text.set(txt)
    except Exception as e:
        result_text.set("Error: " + str(e))
    finally:
        compare_button.config(state=tk.NORMAL)

def on_compare(e1, e2, result_text, compare_button):
    p1 = e1.get().strip()
    p2 = e2.get().strip()
    if not p1 or not p2:
        messagebox.showwarning("Missing files", "Please select two audio files first.")
        return
    # Run in thread to keep UI responsive
    threading.Thread(
        target=run_compare,
        args=(p1, p2, result_text, compare_button),
        daemon=True
    ).start()

def build_ui():
    root = tk.Tk()
    root.title("Voice Matcher — Demo")
    root.geometry("640x320")

    tk.Label(root, text="Voice Matcher — Demo", font=(None, 16, 'bold')).pack(pady=8)

    frame = tk.Frame(root)
    frame.pack(padx=12, pady=8, fill=tk.X)

    tk.Label(frame, text="File 1:").grid(row=0, column=0, sticky='w')
    e1 = tk.Entry(frame, width=60)
    e1.grid(row=0, column=1, padx=6)
    tk.Button(frame, text="Browse", command=lambda: select_file(e1)).grid(row=0, column=2)

    tk.Label(frame, text="File 2:").grid(row=1, column=0, sticky='w')
    e2 = tk.Entry(frame, width=60)
    e2.grid(row=1, column=1, padx=6)
    tk.Button(frame, text="Browse", command=lambda: select_file(e2)).grid(row=1, column=2)

    result_text = tk.StringVar()
    result_text.set("Select two audio files and click Compare.")

    compare_button = tk.Button(
        root, text="Compare", width=20,
        command=lambda: on_compare(e1, e2, result_text, compare_button)
    )
    compare_button.pack(pady=12)

    tk.Label(root, textvariable=result_text, justify='left', anchor='w').pack(padx=12, fill=tk.BOTH)

    root.mainloop()

if __name__ == '__main__':
    build_ui()
