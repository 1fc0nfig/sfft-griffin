#!/usr/bin/env python3
"""
Analyze audio file levels in dBFS (no charts).
"""

import argparse
import numpy as np
from scipy.io import wavfile

def rms_db(x):
    rms = np.sqrt(np.mean(x**2))
    return 20 * np.log10(rms + 1e-12)

def peak_db(x):
    peak = np.max(np.abs(x))
    return 20 * np.log10(peak + 1e-12)

def analyze_audio(audio_path):
    sr, x = wavfile.read(audio_path)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)

    duration = len(x) / sr
    rms_level = rms_db(x)
    peak_level = peak_db(x)
    crest = peak_level - rms_level

    print(f"File: {audio_path}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {duration:.2f} s")
    print(f"Samples: {len(x)}")
    print(f"Mean: {x.mean():.6f}")
    print(f"Std Dev: {x.std():.6f}")
    print(f"RMS Level: {rms_level:.2f} dBFS")
    print(f"Peak Level: {peak_level:.2f} dBFS")
    print(f"Crest Factor: {crest:.2f} dB")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Analyze audio levels in dBFS")
    p.add_argument("audio", help="Path to audio file (WAV)")
    args = p.parse_args()
    analyze_audio(args.audio)
