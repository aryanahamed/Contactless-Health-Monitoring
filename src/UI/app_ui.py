import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np

matplotlib.use("TkAgg")

# Simulate physiological signal
#THIS HAS TO BE REPLACED, IT IS JUST A SIMULATION
def simulate_signal():
    t = np.linspace(0, 10, 500)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.random.randn(len(t))
    return t, signal
#def get_real_signal():
    # Replace with actual signal extraction logic
    # Example: return timestamps, processed_pos_signal
    
   # your filtered POS signal
    #return t, signal


# Simulate HR, BR, HRV
#THIS HAS TO BE REPLACED, IT IS JUST A SIMULATION
def simulate_metrics():
    return {
        "heart_rate": random.randint(60, 100),
        "breathing_rate": random.randint(12, 20),
        "sdnn": round(random.uniform(20, 60), 1),
        "rmssd": round(random.uniform(25, 50), 1),
    }
#def get_real_metrics():
    # Replace this with real-time output of the processing pipeline (pos_processing)
    #return {
       # "heart_rate": computed_hr,
       # "breathing_rate": computed_br,
       # "sdnn": computed_sdnn,
        #"rmssd": computed_rmssd,} #I added the last two values too in the UI

# --- GUI Setup ---
root = tk.Tk()
root.title("rPPG")
root.configure(bg="#1e1e1e")
root.minsize(1100, 700)

# Global flag for updates
update_running = False

# --- Layout Frames ---
#Using dark mode
main_frame = tk.Frame(root, bg="#1e1e1e")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

plot_frame = tk.Frame(main_frame, bg="#1e1e1e")
plot_frame.grid(row=0, column=0, sticky="nws", padx=10, pady=10)

info_frame = tk.Frame(main_frame, bg="#1e1e1e")
info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(side="bottom", pady=20)

# --- Plotting Area ---
plot_canvases = []

def add_plot(parent, title, color, row):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('white')
    ax.set_title(title, color='white')
    t, signal = simulate_signal()
    line, = ax.plot(t, signal, color)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().grid(row=row, column=0, pady=5)
    plot_canvases.append((canvas, line, ax))

add_plot(plot_frame, "Filtered Pulse Signal", 'cyan', 0)
add_plot(plot_frame, "Heart Rate", 'yellow', 1)
add_plot(plot_frame, "Breathing Rate", 'lime', 2)

# --- Metrics ---
metrics_labels = {}

def metric_label(master, key, title, unit=""):
    tk.Label(master, text=title, fg="white", bg="#1e1e1e", font=("Segoe UI", 12)).pack(anchor='w')
    lbl = tk.Label(master, text="", fg="white", bg="#1e1e1e", font=("Segoe UI", 24, "bold"))
    lbl.pack(anchor='w', pady=(0, 10))
    metrics_labels[key] = (lbl, unit)

metric_label(info_frame, "heart_rate", "Heart Rate", "bpm")
metric_label(info_frame, "breathing_rate", "Breathing Rate", "brpm")
metric_label(info_frame, "sdnn", "HRV SDNN", "ms")
metric_label(info_frame, "rmssd", "HRV RMSSD", "ms")

# --- Webcam Placeholder ---
tk.Label(info_frame, text="Live Webcam Feed", fg="white", bg="#1e1e1e", font=("Segoe UI", 12)).pack(anchor='w')
webcam_placeholder = tk.Label(info_frame, bg="black", width=50, height=10)
webcam_placeholder.pack(pady=(0, 20))
#THE PIPELINE ALREADY USE THE WEBCAM? IN THAT CASE:
#import cv2
#from PIL import Image, ImageTk

#cap = cv2.VideoCapture(0)

#def update_webcam_feed():
 #   if not update_running:
  #      return
   # ret, frame = cap.read()
    #if ret:
     #   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #  img = Image.fromarray(frame)
       # imgtk = ImageTk.PhotoImage(image=img)
        #webcam_placeholder.imgtk = imgtk
        #webcam_placeholder.configure(image=imgtk)
    #root.after(30, update_webcam_feed)

# --- Buttons ---
style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", padding=6, relief="flat", background="#000000", foreground="white")
style.map("TButton",
    background=[('active', '#333333')],
    foreground=[('pressed', 'white'), ('active', 'white')]
)

def update_data():
    if not update_running:
        return

    # Update plots
    for canvas, line, ax in plot_canvases:
        t, signal = simulate_signal()
        line.set_data(t, signal)
        ax.relim()
        ax.autoscale_view()
        canvas.draw()

    # Update metrics
    new_metrics = simulate_metrics()
    for key, (lbl, unit) in metrics_labels.items():
        lbl.config(text=f"{new_metrics[key]} {unit}")

    root.after(1000, update_data)
#THIS WHOLE FUNCTION HAS TO BE CHANGHED WITH THIS
#def update_data():
    #if not update_running:
     #   return

    # Update plots with real signal
    #t, signal = get_real_signal()
    #for canvas, line, ax in plot_canvases:
     #   line.set_data(t, signal)
      #  ax.relim()
       # ax.autoscale_view()
        #canvas.draw()

    # Update metrics with real physiological values
    #new_metrics = get_real_metrics()
    #for key, (lbl, unit) in metrics_labels.items():
     #   lbl.config(text=f"{new_metrics[key]} {unit}")

    #root.after(1000, update_data)
#
def start_updates():
    global update_running
    update_running = True
    update_data()

def stop_updates():
    global update_running
    update_running = False

start_btn = ttk.Button(btn_frame, text="Start", command=start_updates)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = ttk.Button(btn_frame, text="Stop", command=stop_updates)
stop_btn.grid(row=0, column=1, padx=10)

# --- Run ---
root.mainloop()

#WHEN THE WINDOW CLOSES, THE WEBCAM HAS TO CLOSE PROPERLY SO WE COULD USE
#def on_close():
    #stop_updates()
    #if cap.isOpened():
     #   cap.release()
    #root.destroy()

#root.protocol("WM_DELETE_WINDOW", on_close)
#sorry for mistakes but without any code was impossible to project properly



