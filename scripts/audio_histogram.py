#!/usr/bin/env python3
"""
Display audio histogram, waveform, and spectrogram.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, get_window

def plot_audio_views(audio_path, bins=100, nfft=1024, hop=256, window='hann', maxfreq=None, 
                    show_histogram=True, show_waveform=True, show_spectrogram=True, log_freq=False, save_spectrogram=False):
    # Read WAV
    sr, x = wavfile.read(audio_path)

    # Mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Normalize to [-1,1] if integer
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)

    # Times
    t = np.arange(len(x)) / sr

    # Determine which plots to show
    plots_to_show = []
    if show_histogram:
        plots_to_show.append('histogram')
    if show_waveform:
        plots_to_show.append('waveform')
    if show_spectrogram:
        plots_to_show.append('spectrogram')
    
    if not plots_to_show:
        print("No plots selected to display!")
        return
    
    # Create subplots based on what to show
    n_plots = len(plots_to_show)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), constrained_layout=True)
    
    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0

    # Histogram
    if show_histogram:
        ax = axes[plot_idx]
        ax.hist(x, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Sample Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Audio Histogram - {audio_path}')
        ax.grid(True, alpha=0.3)

        # Stats
        stats = (
            f'Sample Rate: {sr} Hz\n'
            f'Duration: {len(x)/sr:.2f} s\n'
            f'Samples: {len(x)}\n'
            f'Mean: {x.mean():.4f}\n'
            f'Std Dev: {x.std():.4f}\n'
            f'Min: {x.min():.4f}\n'
            f'Max: {x.max():.4f}'
        )
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontfamily='monospace')
        plot_idx += 1

    # Waveform
    if show_waveform:
        ax = axes[plot_idx]
        ax.plot(t, x, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t[-1] if len(t) else 0)
        plot_idx += 1

    # Spectrogram
    if show_spectrogram:
        ax = axes[plot_idx]
        nperseg = int(nfft)
        noverlap = int(nfft - hop) if hop < nfft else nfft // 2
        win = get_window(window, nperseg, fftbins=True)

        f, tt, Sxx = spectrogram(
            x, fs=sr, window=win, nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling='density', mode='magnitude'
        )
        if maxfreq:
            mask = f <= maxfreq
            f, Sxx = f[mask], Sxx[mask, :]

        Sxx_db = 20 * np.log10(Sxx + 1e-12)  # magnitude dB

        if log_freq:
            # Use log scale for frequency axis
            # Filter out DC component and very low frequencies for better log scale
            f_min = 1.0  # Start from 1 Hz to avoid log(0)
            mask = f >= f_min
            f_log = f[mask]
            Sxx_log = Sxx_db[mask, :]
            
            im = ax.pcolormesh(tt, f_log, Sxx_log, shading='auto')
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (Hz) - Log Scale')
            ax.set_title('Spectrogram (dB) - Log Frequency')
            # Set reasonable frequency limits for audio
            ax.set_ylim(20, min(sr/2, 20000))  # 20 Hz to 20 kHz or Nyquist frequency
        else:
            im = ax.pcolormesh(tt, f, Sxx_db, shading='auto')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Spectrogram (dB)')
        
        ax.set_xlabel('Time (s)')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('dB')

    if save_spectrogram and show_spectrogram:
        # Generate output filename
        import os
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_file = f"{base_name}_spectrogram.png"
        
        # Save the figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Spectrogram saved as: {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Display histogram, waveform, and spectrogram of a WAV file')
    p.add_argument('audio', help='Path to audio file (WAV)')
    p.add_argument('--bins', type=int, default=100, help='Histogram bins (default: 100)')
    p.add_argument('--nfft', type=int, default=1024, help='FFT size for spectrogram (default: 1024)')
    p.add_argument('--hop', type=int, default=256, help='Hop size (default: 256)')
    p.add_argument('--window', default='hann', help='Window type (default: hann)')
    p.add_argument('--maxfreq', type=float, default=None, help='Max frequency to display (Hz)')
    p.add_argument('--no-histogram', action='store_true', help='Hide histogram plot')
    p.add_argument('--no-waveform', action='store_true', help='Hide waveform plot')
    p.add_argument('--no-spectrogram', action='store_true', help='Hide spectrogram plot')
    p.add_argument('--only-histogram', action='store_true', help='Show only histogram')
    p.add_argument('--only-waveform', action='store_true', help='Show only waveform')
    p.add_argument('--only-spectrogram', action='store_true', help='Show only spectrogram')
    p.add_argument('--log-freq', action='store_true', help='Use log scale for frequency axis in spectrogram')
    p.add_argument('--save-spectrogram', action='store_true', help='Save spectrogram as PNG file instead of displaying')
    args = p.parse_args()

    # Determine which plots to show
    show_histogram = not args.no_histogram and not (args.only_waveform or args.only_spectrogram)
    show_waveform = not args.no_waveform and not (args.only_histogram or args.only_spectrogram)
    show_spectrogram = not args.no_spectrogram and not (args.only_histogram or args.only_waveform)
    
    # Handle "only" flags
    if args.only_histogram:
        show_histogram = True
        show_waveform = False
        show_spectrogram = False
    elif args.only_waveform:
        show_histogram = False
        show_waveform = True
        show_spectrogram = False
    elif args.only_spectrogram:
        show_histogram = False
        show_waveform = False
        show_spectrogram = True

    plot_audio_views(args.audio, args.bins, args.nfft, args.hop, args.window, args.maxfreq,
                    show_histogram, show_waveform, show_spectrogram, args.log_freq, args.save_spectrogram)
