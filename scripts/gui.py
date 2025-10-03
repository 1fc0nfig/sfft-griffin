#!/usr/bin/env python3
"""Simple Tkinter UI for tweaking sfft-griffin CLI parameters."""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk
from scipy.io import wavfile
from scipy.signal import spectrogram
import shlex

REPO_ROOT = Path(__file__).resolve().parents[1]

PARAMS = [
    {"name": "duration", "flag": "--duration", "type": "float", "default": 5.0},
    {"name": "sample_rate", "flag": "--sample-rate", "type": "int", "default": 44100},
    {"name": "fft_size", "flag": "--fft-size", "type": "int", "default": 4096},
    {"name": "hop_length", "flag": "--hop-length", "type": "int", "default": ""},
    {"name": "iterations", "flag": "--iterations", "type": "int", "default": 64},
    {"name": "invert", "flag": "--invert", "type": "bool", "default": False},
    {"name": "density", "flag": "--density", "type": "float", "default": 1.0},
    {"name": "sharpness", "flag": "--sharpness", "type": "int", "default": ""},
    {"name": "log_freq", "flag": "--log-freq", "type": "bool", "default": False},
    {"name": "mapping", "flag": "--mapping", "type": "choice", "choices": ["db", "power", "linear"], "default": "power"},
    {"name": "auto_db_range", "flag": "--auto-db-range", "type": "bool", "default": False},
    {"name": "gamma", "flag": "--gamma", "type": "float", "default": 1.0},
    {"name": "scale", "flag": "--scale", "type": "float", "default": 6.0},
    {"name": "db_min", "flag": "--db-min", "type": "float", "default": -120.0},
    {"name": "db_max", "flag": "--db-max", "type": "float", "default": -6.0},
    {"name": "mag_floor", "flag": "--mag-floor", "type": "float", "default": 0.02},
    {"name": "per_frame_norm", "flag": "--per-frame-norm", "type": "bool", "default": False},
    {"name": "silence_threshold", "flag": "--silence-threshold", "type": "float", "default": 0.02},
    {"name": "pct_clip_low", "flag": "--pct-clip-low", "type": "float", "default": 0.0},
    {"name": "pct_clip_high", "flag": "--pct-clip-high", "type": "float", "default": 100.0},
    {"name": "freq_smooth", "flag": "--freq-smooth", "type": "int", "default": 1},
    {"name": "time_smooth", "flag": "--time-smooth", "type": "int", "default": 1},
    {"name": "time_med", "flag": "--time-med", "type": "int", "default": 1},
    {"name": "bandpass_low", "flag": "--bandpass-low", "type": "float", "default": 0.0},
    {"name": "bandpass_high", "flag": "--bandpass-high", "type": "float", "default": ""},
    {"name": "gate_percentile", "flag": "--gate-percentile", "type": "float", "default": 70.0},
    {"name": "gate_reduction", "flag": "--gate-reduction", "type": "float", "default": 0.1},
    {"name": "gl_momentum", "flag": "--gl-momentum", "type": "float", "default": 0.2},
    {"name": "preemph", "flag": "--preemph", "type": "float", "default": 0.0},
    {"name": "post_lp", "flag": "--post-lp", "type": "float", "default": 0.0},
    {"name": "fade_in_pct", "flag": "--fade-in-pct", "type": "float", "default": 0.0},
    {"name": "fade_out_pct", "flag": "--fade-out-pct", "type": "float", "default": 0.0},
    {"name": "target_rms", "flag": "--target-rms", "type": "float", "default": 0.22},
    {"name": "target_rms_auto", "flag": "--target-rms-auto", "type": "bool", "default": False},
    {"name": "limiter_ceiling", "flag": "--limiter-ceiling", "type": "float", "default": 0.98},
    {"name": "softclip_knee", "flag": "--softclip-knee", "type": "float", "default": 0.0},
    {"name": "softclip_slope", "flag": "--softclip-slope", "type": "float", "default": 0.15},
    {"name": "no_normalize", "flag": "--no-normalize", "type": "bool", "default": False},
    {"name": "output_gain", "flag": "--output-gain", "type": "float", "default": 1.0},
    {"name": "float32_out", "flag": "--float32-out", "type": "bool", "default": True},
]

IMAGE_PREVIEW_MAX_SIZE = (400, 300)
SPECTROGRAM_SIZE = (400, 200)
SETTINGS_VERSION = 3

class GriffinUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("sfft-griffin parameter explorer")
        self.geometry("900x700")

        self.image_path = tk.StringVar()
        self.image_path.trace_add("write", self._on_image_path_change)
        self.output_path = tk.StringVar()
        self.output_path.trace_add("write", self._on_output_path_change)

        self.param_vars = {}
        self._image_preview_photo = None
        self._spectrogram_preview_photo = None
        self._loading_settings = False
        self._console_visible = True
        self._build_layout()
        self._update_previews()
        self._load_settings()

    def _build_layout(self):
        self.geometry("1100x750")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(self)
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)
        side_panel.rowconfigure(3, weight=1)

        # Paths
        paths_frame = ttk.LabelFrame(side_panel, text="Paths")
        paths_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        paths_frame.columnconfigure(1, weight=1)

        ttk.Label(paths_frame, text="Input image:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        image_entry = ttk.Entry(paths_frame, textvariable=self.image_path, width=28)
        image_entry.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(paths_frame, text="Browse", command=self._choose_image).grid(row=0, column=2, padx=(5, 0), pady=2)

        ttk.Label(paths_frame, text="Output WAV:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        output_entry = ttk.Entry(paths_frame, textvariable=self.output_path, width=28)
        output_entry.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Button(paths_frame, text="Browse", command=self._choose_output).grid(row=1, column=2, padx=(5, 0), pady=2)

        # Parameters
        params_container = ttk.LabelFrame(side_panel, text="Parameters")
        params_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        params_container.columnconfigure(0, weight=1)
        params_container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(params_container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        scrollable = ttk.Frame(canvas)
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        params_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(params_window, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        for idx, param in enumerate(PARAMS):
            row = ttk.Frame(scrollable)
            row.grid(row=idx, column=0, sticky="ew", pady=2, padx=2)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=param["flag"], width=18).grid(row=0, column=0, sticky=tk.W)

            if param["type"] == "bool":
                var = tk.BooleanVar(value=param["default"])
                cb = ttk.Checkbutton(row, variable=var)
                cb.grid(row=0, column=1, sticky=tk.W)
            elif param["type"] == "choice":
                var = tk.StringVar(value=param["default"])
                combo = ttk.Combobox(row, textvariable=var, values=param["choices"], state="readonly", width=12)
                combo.grid(row=0, column=1, sticky=tk.W)
            else:
                default = "" if param["default"] == "" else str(param["default"])
                var = tk.StringVar(value=default)
                entry = ttk.Entry(row, textvariable=var, width=15)
                entry.grid(row=0, column=1, sticky="ew")
            self.param_vars[param["name"]] = (param, var)

        # Action buttons
        button_frame = ttk.Frame(side_panel)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 0))
        button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(button_frame, text="Run", command=self._run_conversion)
        self.run_button.grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Reset defaults", command=self._reset_defaults).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="Open output folder", command=self._open_output_folder).grid(row=0, column=2)

        self.progress = ttk.Progressbar(button_frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        # Console output (collapsible)
        console_container = ttk.Frame(side_panel)
        console_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=(5, 10))
        console_container.columnconfigure(0, weight=1)
        console_container.rowconfigure(1, weight=1)

        console_header = ttk.Frame(console_container)
        console_header.grid(row=0, column=0, sticky="ew")
        ttk.Label(console_header, text="Console output").pack(side=tk.LEFT)
        self.console_toggle_btn = ttk.Button(console_header, text="Hide", width=8, command=self._toggle_console)
        self.console_toggle_btn.pack(side=tk.RIGHT)

        self.console_frame = ttk.Frame(console_container)
        self.console_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        self.console_frame.columnconfigure(0, weight=1)
        self.console_frame.rowconfigure(0, weight=1)

        console_scroll = ttk.Scrollbar(self.console_frame, orient="vertical")
        self.output_text = tk.Text(self.console_frame, height=12, wrap="none", yscrollcommand=console_scroll.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        console_scroll.config(command=self.output_text.yview)
        console_scroll.grid(row=0, column=1, sticky="ns")

        # Preview column
        preview_frame = ttk.Frame(self)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        image_frame = ttk.LabelFrame(preview_frame, text="Source image")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_preview_label = tk.Label(
            image_frame,
            relief=tk.SUNKEN,
            bd=1,
            bg="white",
            anchor="center",
            justify=tk.CENTER,
        )
        self.image_preview_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.image_preview_label.configure(text="No image selected")

        spectrogram_frame = ttk.LabelFrame(preview_frame, text="Spectrogram")
        spectrogram_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        spectrogram_frame.columnconfigure(0, weight=1)
        spectrogram_frame.rowconfigure(0, weight=1)
        self.spectrogram_preview_label = tk.Label(
            spectrogram_frame,
            relief=tk.SUNKEN,
            bd=1,
            bg="white",
            anchor="center",
            justify=tk.CENTER,
        )
        self.spectrogram_preview_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.spectrogram_preview_label.configure(text="Spectrogram unavailable")

    def _choose_image(self):
        file = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if file:
            self.image_path.set(file)
            if not self.output_path.get():
                out = Path(file).with_suffix(".wav")
                self.output_path.set(str(out))
            self._persist_settings()

    def _choose_output(self):
        file = filedialog.asksaveasfilename(title="Output WAV", defaultextension=".wav", filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if file:
            self.output_path.set(file)
            self._persist_settings()

    def _on_image_path_change(self, *_):
        self._update_previews()
        self._persist_settings()

    def _on_output_path_change(self, *_):
        self._update_spectrogram_preview()
        self._persist_settings()

    def _build_command(self):
        image = self.image_path.get().strip()
        output = self.output_path.get().strip()
        if not image or not output:
            messagebox.showerror("Missing paths", "Please select both input image and output WAV paths.")
            return None

        cmd = ["cargo", "run", "--release", "--", "--image", image, "--output", output]
        for name, (meta, var) in self.param_vars.items():
            if meta["type"] == "bool":
                value = bool(var.get())
                default = meta.get("default")
                if isinstance(default, bool) and value == default:
                    continue
                if value:
                    cmd.append(meta["flag"])
                elif meta.get("neg_flag"):
                    cmd.append(meta["neg_flag"])
            else:
                text = var.get().strip()
                if text == "":
                    continue
                cmd.extend([meta["flag"], text])
        return cmd

    def _run_conversion(self):
        cmd = self._build_command()
        if not cmd:
            return
        self.output_text.delete("1.0", tk.END)
        pretty_cmd = shlex.join(cmd)
        self._append_output_line(f"$ {pretty_cmd}\n")
        self._append_output_line("Starting conversion...\n")
        self.progress.start()
        self.run_button.state(["disabled"])
        self._set_spectrogram_placeholder("Waiting for audio...")

        def worker():
            proc = None
            returncode: int | None = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None

                for line in iter(proc.stdout.readline, ""):
                    if line:
                        self.after(0, self._append_output_line, line)
                proc.wait()
                returncode = proc.returncode
                if returncode != 0:
                    self.after(
                        0,
                        lambda: messagebox.showerror(
                            "Conversion failed", f"Command exited with {returncode}"
                        ),
                    )
            except FileNotFoundError:
                self.after(0, lambda: messagebox.showerror("Error", "Could not run cargo. Ensure Rust toolchain is installed."))
                self.after(0, lambda: self._append_output_line("Error: cargo executable not found\n"))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
                self.after(0, lambda: self._append_output_line(f"Error: {exc}\n"))
            finally:
                self.after(0, self._on_run_finished, returncode)

        threading.Thread(target=worker, daemon=True).start()
        self._persist_settings()

    def _on_run_finished(self, returncode: int | None) -> None:
        self.progress.stop()
        self.run_button.state(["!disabled"])
        if returncode == 0:
            self._append_output_line("✔ Conversion finished successfully\n")
            self._update_spectrogram_preview()
        else:
            if returncode is not None:
                self._append_output_line(f"✖ Conversion failed with code {returncode}\n")
            self._set_spectrogram_placeholder("Spectrogram unavailable")
        self._persist_settings()

    def _append_output_line(self, line: str) -> None:
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)

    def _toggle_console(self) -> None:
        self._console_visible = not self._console_visible
        if self._console_visible:
            self.console_frame.grid()
            self.console_toggle_btn.configure(text="Hide")
        else:
            self.console_frame.grid_remove()
            self.console_toggle_btn.configure(text="Show")

    def _reset_defaults(self):
        for param, var in self.param_vars.values():
            if param["type"] == "bool":
                var.set(param["default"])
            elif param["type"] == "choice":
                var.set(param["default"])
            else:
                var.set("" if param["default"] == "" else str(param["default"]))
        self.output_text.delete("1.0", tk.END)
        self.progress.stop()
        self._persist_settings()

    def _open_output_folder(self):
        output = self.output_path.get().strip()
        if not output:
            messagebox.showinfo("No output", "Output path is empty.")
            return
        folder = Path(output).resolve().parent
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", folder])
        elif os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", folder])

    def _persist_settings(self, *_):
        if getattr(self, "_loading_settings", False):
            return
        data = {
            "image": self.image_path.get(),
            "output": self.output_path.get(),
            "params": {
                name: (var.get() if isinstance(var, tk.StringVar) else bool(var.get()))
                for name, (_, var) in self.param_vars.items()
            },
            "version": SETTINGS_VERSION,
        }
        try:
            settings_path = REPO_ROOT / "scripts" / ".gui_settings.json"
            settings_path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def _load_settings(self):
        settings_path = REPO_ROOT / "scripts" / ".gui_settings.json"
        if not settings_path.exists():
            return
        try:
            self._loading_settings = True
            data = json.loads(settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._loading_settings = False
            return
        if data.get("version") != SETTINGS_VERSION:
            # Reset to current defaults but keep remembered paths.
            self.image_path.set(data.get("image", ""))
            self.output_path.set(data.get("output", ""))
            self._loading_settings = False
            self._reset_defaults()
            return

        self.image_path.set(data.get("image", ""))
        self.output_path.set(data.get("output", ""))
        for name, value in data.get("params", {}).items():
            if name in self.param_vars:
                meta, var = self.param_vars[name]
                if meta["type"] == "bool":
                    var.set(bool(value))
                else:
                    var.set(str(value))
        self._loading_settings = False
        self._update_previews()

    def _update_previews(self):
        self._update_image_preview()
        self._update_spectrogram_preview()

    def _update_image_preview(self):
        path = self.image_path.get().strip()
        if not path:
            self._set_image_placeholder("No image selected")
            return
        image_file = Path(path).expanduser()
        if not image_file.exists():
            self._set_image_placeholder("Image not found")
            return
        try:
            with Image.open(image_file) as original:
                source_image = original.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._set_image_placeholder(f"Error loading image:\n{exc}")
            return

        preview = ImageOps.contain(source_image, IMAGE_PREVIEW_MAX_SIZE)
        self._image_preview_photo = ImageTk.PhotoImage(preview)
        self.image_preview_label.configure(image=self._image_preview_photo, text="")

    def _update_spectrogram_preview(self):
        output = self.output_path.get().strip()
        if not output:
            self._set_spectrogram_placeholder("No output selected")
            return
        audio_file = Path(output).expanduser()
        if not audio_file.exists():
            self._set_spectrogram_placeholder("Spectrogram unavailable\nRun conversion first")
            return
        try:
            spectrogram_image = self._render_spectrogram_image(audio_file)
        except Exception as exc:  # noqa: BLE001
            self._set_spectrogram_placeholder(f"Error loading audio:\n{exc}")
            return

        self._spectrogram_preview_photo = ImageTk.PhotoImage(spectrogram_image)
        self.spectrogram_preview_label.configure(image=self._spectrogram_preview_photo, text="")

    def _set_image_placeholder(self, text: str) -> None:
        self._image_preview_photo = None
        self.image_preview_label.configure(text=text, image="")

    def _set_spectrogram_placeholder(self, text: str) -> None:
        self._spectrogram_preview_photo = None
        self.spectrogram_preview_label.configure(text=text, image="")

    def _render_spectrogram_image(self, audio_path: Path) -> Image.Image:
        sample_rate, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            max_val = float(max(abs(info.min), abs(info.max)))
            if max_val == 0:
                raise ValueError("Invalid WAV data")
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)

        if data.size == 0:
            raise ValueError("Empty audio file")

        nfft = 1024
        hop = 256
        window = "hann"
        freqs, times, spectrum = spectrogram(
            data,
            fs=sample_rate,
            nperseg=nfft,
            noverlap=nfft - hop,
            window=window,
            scaling="density",
            mode="magnitude",
        )

        spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
        peak = spectrum_db.max()
        floor = peak - 80.0
        normalized = (spectrum_db - floor) / max(peak - floor, 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Flip vertically so low freqs at bottom
        normalized = normalized[::-1, :]
        grayscale = (normalized * 255).astype(np.uint8)
        image = Image.fromarray(grayscale, mode="L").convert("RGB")
        image = ImageOps.contain(image, SPECTROGRAM_SIZE)
        return image


if __name__ == "__main__":
    app = GriffinUI()
    app.mainloop()
