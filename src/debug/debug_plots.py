import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

PLOT_PPG = False
PLOT_BR = False
PLOT_RGB_DIAG = False

def setup_debug_figures():
    figs = [None, None, None, None, None]
    if PLOT_PPG or PLOT_BR:
        plt.ion()
        signal_figure, (signal_ax, br_ax) = plt.subplots(2, 1, figsize=(10, 8))
        signal_figure.suptitle('Signal Processing Debug')
        signal_ax.set_title('PPG Signal and Detected Peaks')
        signal_ax.set_ylabel('Amplitude')
        signal_ax.set_xlabel('Sample')
        br_ax.set_title('Breathing Signal')
        br_ax.set_ylabel('Amplitude')
        br_ax.set_xlabel('Sample')
        plt.tight_layout()
        figs[0:3] = [signal_figure, signal_ax, br_ax]
    if PLOT_RGB_DIAG:
        plt.ion()
        rgb_diag_fig, rgb_diag_axs = plt.subplots(3, 1, figsize=(10, 8))
        rgb_diag_fig.suptitle('RGB Diagnostics')
        rgb_diag_axs[0].set_ylabel('Raw RGB')
        rgb_diag_axs[1].set_ylabel('Normalized RGB')
        rgb_diag_axs[2].set_ylabel('Detrended RGB')
        rgb_diag_axs[2].set_xlabel('Sample')
        plt.tight_layout()
        figs[3:5] = [rgb_diag_fig, rgb_diag_axs]
    return figs[0], figs[1], figs[2], figs[3], figs[4]

def plot_ppg_signal(signal_ax, best_filt, peaks_tuple=None):
    signal_ax.clear()
    signal_ax.plot(best_filt, 'b-', label='PPG Signal')
    if peaks_tuple and len(peaks_tuple[0]) > 0:
        signal_ax.plot(peaks_tuple[0], best_filt[peaks_tuple[0]], 'ro', label='Detected Peaks')
    signal_ax.set_title('PPG Signal and Detected Peaks')
    signal_ax.set_ylabel('Amplitude')
    signal_ax.set_xlabel('Sample')
    signal_ax.legend()
    signal_ax.figure.canvas.draw_idle()
    plt.pause(0.01)

def plot_breathing_signal(br_ax, br_signal):
    br_ax.clear()
    br_ax.plot(br_signal, 'g-', label='Breathing Signal')
    br_ax.set_title('Breathing Signal')
    br_ax.legend()
    br_ax.figure.canvas.draw_idle()
    plt.pause(0.01)

def plot_rgb_diagnostics(rgb_diag_axs, best_rgb):
    rgb_diag_axs[0].clear()
    rgb_diag_axs[0].plot(best_rgb[:, 0], label='Raw R')
    rgb_diag_axs[0].plot(best_rgb[:, 1], label='Raw G')
    rgb_diag_axs[0].plot(best_rgb[:, 2], label='Raw B')
    rgb_diag_axs[0].set_title('Raw Mean RGB')
    rgb_diag_axs[0].legend()
    mean_rgb = np.mean(best_rgb, axis=0)
    norm_rgb = best_rgb / mean_rgb
    rgb_diag_axs[1].clear()
    rgb_diag_axs[1].plot(norm_rgb[:, 0], label='Norm R')
    rgb_diag_axs[1].plot(norm_rgb[:, 1], label='Norm G')
    rgb_diag_axs[1].plot(norm_rgb[:, 2], label='Norm B')
    rgb_diag_axs[1].set_title('Normalized RGB')
    rgb_diag_axs[1].legend()
    detrended_rgb = scipy.signal.detrend(norm_rgb, axis=0)
    rgb_diag_axs[2].clear()
    rgb_diag_axs[2].plot(detrended_rgb[:, 0], label='Detrended R')
    rgb_diag_axs[2].plot(detrended_rgb[:, 1], label='Detrended G')
    rgb_diag_axs[2].plot(detrended_rgb[:, 2], label='Detrended B')
    rgb_diag_axs[2].set_title('Detrended RGB')
    rgb_diag_axs[2].legend()
    rgb_diag_axs[2].figure.canvas.draw_idle()
    plt.pause(0.01)
