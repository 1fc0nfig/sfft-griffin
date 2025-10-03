use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use image::{GenericImageView, ImageBuffer, Luma};
use rand::Rng;
use rustfft::{num_complex::Complex64, FftPlanner};
use std::f64::consts::PI;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
enum MappingType {
    Linear,
    Power,
    Db,
}

impl std::str::FromStr for MappingType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(MappingType::Linear),
            "power" => Ok(MappingType::Power),
            "db" => Ok(MappingType::Db),
            _ => Err(format!("Invalid mapping '{}'. Use: linear, power, db", s)),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Image to Audio via Auto-Tuned Griffin-Lim")]
struct Args {
    #[arg(short, long)]
    image: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long, default_value_t = 5.0)]
    duration: f64,
    #[arg(short, long, default_value_t = 44100)]
    sample_rate: u32,
    #[arg(short, long, default_value_t = 4096)]
    fft_size: usize,
    #[arg(short = 'l', long)]
    hop_length: Option<usize>,
    #[arg(short = 'n', long, default_value_t = 64)]
    iterations: usize,

    // Image processing
    #[arg(long, default_value_t = false)]
    invert: bool,
    #[arg(long, default_value_t = 1.0)]
    density: f64,
    #[arg(long)]
    sharpness: Option<u8>,
    #[arg(long, default_value_t = false)]
    log_freq: bool,

    // Magnitude mapping (power by default for robustness)
    #[arg(short = 'm', long, default_value = "power")]
    mapping: String,
    #[arg(long, default_value_t = false)]
    auto_db_range: bool,
    #[arg(short = 'g', long, default_value_t = 1.0)]
    gamma: f64,
    #[arg(long, default_value_t = 6.0)]
    scale: f64,
    #[arg(long, default_value_t = -120.0, allow_hyphen_values = true)]
    db_min: f64,
    #[arg(long, default_value_t = -6.0, allow_hyphen_values = true)]
    db_max: f64,
    #[arg(long, default_value_t = 0.02)]
    mag_floor: f64,
    #[arg(long, default_value_t = false)]
    per_frame_norm: bool,
    #[arg(long, default_value_t = 0.02)]
    silence_threshold: f64,

    // Spectrogram conditioning (moderate to preserve contrast)
    #[arg(long, default_value_t = 0.0)]
    pct_clip_low: f64,
    #[arg(long, default_value_t = 100.0)]
    pct_clip_high: f64,
    #[arg(long, default_value_t = 1)]
    freq_smooth: usize,
    #[arg(long, default_value_t = 1)]
    time_smooth: usize,
    #[arg(long, default_value_t = 1)]
    time_med: usize,
    #[arg(long, default_value_t = 0.0)]
    bandpass_low: f64,
    #[arg(long)]
    bandpass_high: Option<f64>,

    // Spectral gating (very aggressive - nearly silence black regions)
    #[arg(long, default_value_t = 70.0)]
    gate_percentile: f64,
    #[arg(long, default_value_t = 0.1)]
    gate_reduction: f64,

    // Fast Griffin-Lim
    #[arg(long, default_value_t = 0.2)]
    gl_momentum: f64,

    // Post-processing
    #[arg(long, default_value_t = 0.0)]
    preemph: f64,
    #[arg(long, default_value_t = 0.0)]
    post_lp: f64,
    #[arg(long, default_value_t = 0.0)]
    fade_in_pct: f64,
    #[arg(long, default_value_t = 0.0)]
    fade_out_pct: f64,

    // Normalization (moderate target to avoid boosting noise)
    #[arg(long, default_value_t = 0.22)]
    target_rms: f64,
    #[arg(long, default_value_t = false)]
    target_rms_auto: bool,
    #[arg(long, default_value_t = 0.98)]
    limiter_ceiling: f64,
    #[arg(long, default_value_t = 0.0)]
    softclip_knee: f64,
    #[arg(long, default_value_t = 0.15)]
    softclip_slope: f64,
    #[arg(long, default_value_t = false)]
    no_normalize: bool,
    #[arg(long, default_value_t = 1.0)]
    output_gain: f64,
    #[arg(long, default_value_t = true)]
    float32_out: bool,
}

fn apply_sharpness_preset(args: &mut Args, sharpness: u8) {
    // Map sharpness 1-10 to parameter ranges
    // 1 = very soft, 5 = balanced, 10 = maximum sharp
    let s = sharpness as f64;

    // Smoothing: 1→21, 5→5, 10→1
    let smooth_freq = (22.0 - 2.0 * s).round() as usize;
    let smooth_time = (22.0 - 2.0 * s).round() as usize;
    let smooth_med = ((12.0 - 1.0 * s).round() as usize).max(1);

    // Hop length: 1→fft/2, 5→fft/4, 10→fft/8
    // We'll compute this as a divisor
    let hop_divisor = if sharpness <= 5 {
        2.0 + 0.4 * (5.0 - s) // 5→4, 1→5.6 (~fft/2)
    } else {
        4.0 + 0.8 * (s - 5.0) // 5→4, 10→8
    };

    args.freq_smooth = smooth_freq;
    args.time_smooth = smooth_time;
    args.time_med = smooth_med;
    args.hop_length = Some((args.fft_size as f64 / hop_divisor).round() as usize);

    println!(
        "Sharpness preset: {} (smooth=[{},{},{}], hop=fft/{:.1})",
        sharpness, smooth_freq, smooth_time, smooth_med, hop_divisor
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();

    // Apply sharpness preset if specified
    if let Some(sharpness) = args.sharpness {
        if sharpness < 1 || sharpness > 10 {
            return Err("Sharpness must be between 1 and 10".into());
        }
        apply_sharpness_preset(&mut args, sharpness);
    }

    // Validate
    if args.fft_size == 0 || !args.fft_size.is_power_of_two() {
        return Err(format!("FFT size must be power of 2, got {}", args.fft_size).into());
    }
    let hop_length = args.hop_length.unwrap_or(args.fft_size / 4);
    if hop_length == 0 || hop_length >= args.fft_size {
        return Err("Invalid hop length".into());
    }
    if args.duration <= 0.0 {
        return Err("Duration must be positive".into());
    }
    if args.preemph < 0.0 || args.preemph >= 1.0 {
        return Err("Pre-emphasis must be in [0, 1)".into());
    }

    let mapping_type: MappingType = args.mapping.parse()?;

    // Auto bandpass_high if not specified (allow full range)
    let nyquist = args.sample_rate as f64 / 2.0;
    let bandpass_high = args.bandpass_high.unwrap_or(0.95 * nyquist);

    println!("=== Auto-Tuned Griffin-Lim Image→Audio ===");
    println!(
        "DSP: sr={} Hz, FFT={}, hop={}, dur={:.1}s, iters={}",
        args.sample_rate, args.fft_size, hop_length, args.duration, args.iterations
    );
    println!(
        "Mapping: {:?}{}",
        mapping_type,
        if args.auto_db_range && matches!(mapping_type, MappingType::Db) {
            " (auto range from histogram)"
        } else {
            ""
        }
    );
    println!(
        "Conditioning: pct=[{:.0}-{:.0}%], smooth=[freq={}, time={}], median={}, per-frame-norm={}",
        args.pct_clip_low,
        args.pct_clip_high,
        args.freq_smooth,
        args.time_smooth,
        args.time_med,
        args.per_frame_norm
    );
    println!(
        "Bandpass: {:.0}-{:.0} Hz, Gate: p{:.0}%/×{:.2}",
        args.bandpass_low, bandpass_high, args.gate_percentile, args.gate_reduction
    );
    println!(
        "Output: {}",
        if args.float32_out { "float32" } else { "int16" }
    );

    // Calculate frames
    let target_samples = (args.duration * args.sample_rate as f64).round() as usize;
    let num_frames = if target_samples >= args.fft_size {
        (target_samples - args.fft_size) / hop_length + 1
    } else {
        return Err("Duration too short".into());
    };
    let _expected_samples = (num_frames - 1) * hop_length + args.fft_size;

    // Load and resize image
    println!("\nLoading: {:?}", args.image);
    let img = image::open(&args.image)?;
    let (w, h) = img.dimensions();

    // Convert to grayscale (ignore alpha in luminance but respect transparency when present)
    let rgba = img.to_rgba32f();
    let mut gray: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let pixel = rgba.get_pixel(x, y);
            let (r, g, b, a) = (pixel[0], pixel[1], pixel[2], pixel[3]);
            let mut luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            luminance *= a.max(0.0).min(1.0);
            if args.invert {
                luminance = 1.0 - luminance;
            }
            gray.put_pixel(x, y, Luma([luminance.clamp(0.0, 1.0)]));
        }
    }

    if args.invert {
        println!("  Image inverted (black ↔ white)");
    }

    if (args.density - 1.0).abs() > f64::EPSILON {
        for pix in gray.pixels_mut() {
            let v = pix.0[0].clamp(0.0, 1.0);
            pix.0[0] = v.powf(args.density as f32);
        }
        println!("  Density applied: {:.2}", args.density);
    }

    let num_freq_bins = args.fft_size / 2 + 1;
    let freq_per_bin = args.sample_rate as f64 / args.fft_size as f64;
    let min_freq = freq_per_bin.max(1.0);
    let max_freq = (args.sample_rate as f64 / 2.0).max(min_freq * (1.0 + 1e-6));
    let log_min = min_freq.ln();
    let log_max = max_freq.ln();
    let log_span = (log_max - log_min).max(f64::EPSILON);
    let width = if w > 1 { (w - 1) as f64 } else { 0.0 };
    let height = if h > 1 { (h - 1) as f64 } else { 0.0 };

    let mut raw_vals = vec![vec![0.0f64; num_frames]; num_freq_bins];
    let mut raw_flat = Vec::with_capacity(num_frames * num_freq_bins);

    for y in 0..num_freq_bins {
        let freq = y as f64 * freq_per_bin;
        let freq_ratio = if args.log_freq {
            if freq <= min_freq {
                0.0
            } else if freq >= max_freq {
                1.0
            } else {
                ((freq.ln() - log_min) / log_span).clamp(0.0, 1.0)
            }
        } else if num_freq_bins > 1 {
            y as f64 / (num_freq_bins as f64 - 1.0)
        } else {
            0.0
        };
        let src_y = (1.0 - freq_ratio) * height;

        for x in 0..num_frames {
            let time_ratio = if num_frames > 1 {
                x as f64 / (num_frames as f64 - 1.0)
            } else {
                0.0
            };
            let src_x = time_ratio * width;
            let px = sample_bilinear(&gray, src_x, src_y);
            raw_vals[y][x] = px;
            raw_flat.push(px);
        }
    }

    let mut auto_db_norm: Option<(f64, f64)> = None;
    if args.auto_db_range && matches!(mapping_type, MappingType::Db) && !raw_flat.is_empty() {
        raw_flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = raw_flat.len();
        let percentile = |q: f64| -> f64 {
            let idx = ((q / 100.0).clamp(0.0, 1.0) * (len as f64 - 1.0)).round() as usize;
            raw_flat[idx]
        };
        let low = percentile(1.0);
        let high = percentile(99.0);
        if high > low + f64::EPSILON {
            auto_db_norm = Some((low, high));
            println!(
                "  Auto dB mapping: norm[{:.3}, {:.3}] → [{:.1}, {:.1}] dB",
                low, high, args.db_min, args.db_max
            );
        } else {
            println!("  Auto dB mapping skipped (degenerate histogram)");
        }
    }

    let mut magnitude = vec![vec![0.0f64; num_frames]; num_freq_bins];
    for y in 0..num_freq_bins {
        for x in 0..num_frames {
            let mag = pixel_to_magnitude(
                raw_vals[y][x],
                mapping_type,
                args.gamma,
                args.scale,
                args.db_min,
                args.db_max,
                auto_db_norm,
            );
            magnitude[y][x] = mag.max(args.mag_floor);
        }
    }

    // Per-frame normalize
    if args.per_frame_norm {
        println!("\nApplying per-frame normalization...");
        per_frame_normalize(&mut magnitude, args.mag_floor);
    }

    println!("\n=== Conditioning ===");

    // Percentile clamp
    let (low_val, high_val) =
        percentile_clamp(&mut magnitude, args.pct_clip_low, args.pct_clip_high);
    println!("Percentile: [{:.6}, {:.6}]", low_val, high_val);

    // Smoothing
    if args.freq_smooth > 1 {
        smooth_along_freq(&mut magnitude, args.freq_smooth);
    }
    if args.time_smooth > 1 {
        smooth_along_time(&mut magnitude, args.time_smooth);
    }
    if args.time_med > 1 {
        median_along_time(&mut magnitude, args.time_med);
    }
    println!(
        "Smoothed: freq={}, time={}, median={}",
        args.freq_smooth, args.time_smooth, args.time_med
    );

    // Bandpass
    let zeroed = apply_bandpass(
        &mut magnitude,
        args.bandpass_low,
        bandpass_high,
        args.sample_rate,
        args.fft_size,
    );
    println!(
        "Bandpass: {:.0}-{:.0} Hz ({} bins zeroed)",
        args.bandpass_low, bandpass_high, zeroed
    );

    // Spectral gate
    let gated_pct = spectral_gate(&mut magnitude, args.gate_percentile, args.gate_reduction);
    println!("Gate: {:.1}% bins reduced", gated_pct);

    // Compression based on sharpness (if set)
    let compress_strength = if let Some(sharpness) = args.sharpness {
        // 1→0.5 (heavy), 5→0.1, 10→0.0 (none)
        ((11.0 - sharpness as f64) * 0.05).max(0.0)
    } else {
        0.1 // Default
    };

    if compress_strength > 0.0 {
        for row in magnitude.iter_mut() {
            for v in row.iter_mut() {
                if *v > 0.01 {
                    let compressed = (*v).powf(1.0 - compress_strength);
                    *v = compressed;
                }
            }
        }
        println!("Magnitude compression: strength={:.2}", compress_strength);
    } else {
        println!("Magnitude compression: disabled (maximum sharpness)");
    }

    // Higher ceiling to allow more energy
    let mag_ceiling = 4.0;
    for row in magnitude.iter_mut() {
        for v in row.iter_mut() {
            if *v > mag_ceiling {
                *v = mag_ceiling;
            }
        }
    }
    println!("Magnitude ceiling: {:.2}", mag_ceiling);

    // Stats
    let mags: Vec<f64> = magnitude.iter().flat_map(|r| r.iter().copied()).collect();
    let min_mag = mags.iter().copied().fold(f64::INFINITY, f64::min);
    let max_mag = mags.iter().copied().fold(0.0f64, f64::max);
    let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
    println!(
        "Range: [{:.6}, {:.6}], Mean: {:.6}",
        min_mag, max_mag, mean_mag
    );

    // Fast Griffin-Lim
    println!("\n=== Griffin-Lim ({} iterations) ===", args.iterations);
    let audio = fast_griffin_lim(
        &magnitude,
        args.fft_size,
        hop_length,
        args.iterations,
        args.gl_momentum,
        args.preemph,
    )?;

    println!("\n=== Post-Processing ===");
    let mut audio_proc = audio.clone();

    // De-emphasis
    if args.preemph > 0.0 {
        de_emphasis(&mut audio_proc, args.preemph);
    }

    // DC block
    dc_block(&mut audio_proc, args.sample_rate);

    // Low-pass
    if args.post_lp > 0.0 {
        one_pole_lp(&mut audio_proc, args.post_lp, args.sample_rate);
    }

    // Fade in/out
    if args.fade_in_pct > 0.0 || args.fade_out_pct > 0.0 {
        apply_fades(&mut audio_proc, args.fade_in_pct, args.fade_out_pct);
        println!(
            "Fades: in={:.1}%, out={:.1}%",
            args.fade_in_pct * 100.0,
            args.fade_out_pct * 100.0
        );
    }

    let rms_raw = compute_rms(&audio_proc);
    let peak_raw = audio_proc.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    let rms_db = if rms_raw > 1e-10 {
        20.0 * rms_raw.log10()
    } else {
        -100.0
    };
    println!(
        "Raw: RMS={:.6} ({:.1} dBFS), Peak={:.6}",
        rms_raw, rms_db, peak_raw
    );

    // Clip extreme peaks before normalization
    let crest_raw = if rms_raw > 1e-10 {
        peak_raw / rms_raw
    } else {
        1.0
    };
    if crest_raw > 10.0 {
        println!("Crest factor: {:.1}× - clipping outlier peaks", crest_raw);
        // Hard clip peaks above 2× RMS (removes impulses without boosting floor)
        let clip_threshold = rms_raw * 2.0;
        let mut clipped_count = 0;
        for s in audio_proc.iter_mut() {
            if s.abs() > clip_threshold {
                *s = s.signum() * clip_threshold;
                clipped_count += 1;
            }
        }
        let rms_after_clip = compute_rms(&audio_proc);
        let peak_after_clip = audio_proc.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        let crest_after_clip = if rms_after_clip > 1e-10 {
            peak_after_clip / rms_after_clip
        } else {
            1.0
        };
        println!(
            "  Clipped {} samples → RMS={:.6}, Peak={:.6}, Crest={:.1}×",
            clipped_count, rms_after_clip, peak_after_clip, crest_after_clip
        );
    }

    // Normalize
    let audio_final = if args.no_normalize {
        println!("Normalization: DISABLED");
        audio_proc
    } else {
        let mut target_rms = args.target_rms;
        if args.target_rms_auto {
            // Predict peak if we hit target RMS; scale target to keep 0.95 * ceiling margin
            let pred_peak = if rms_raw > 1e-12 {
                peak_raw * (target_rms / rms_raw)
            } else {
                0.0
            };
            let safe_peak = args.limiter_ceiling * 0.95;
            if pred_peak > safe_peak && pred_peak > 0.0 {
                let scale = safe_peak / pred_peak;
                let new_target = (target_rms * scale).clamp(0.05, 0.5); // keep reasonable range
                println!(
                    "Auto RMS: {:.3} → {:.3} to respect peak ceiling",
                    target_rms, new_target
                );
                target_rms = new_target;
            }
        }

        let mut audio_norm = audio_proc.clone();
        normalize_audio(
            &mut audio_norm,
            target_rms,
            args.limiter_ceiling,
            args.softclip_knee,
            args.softclip_slope,
        );

        let rms_norm = compute_rms(&audio_norm);
        let peak_norm = audio_norm.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        let rms_norm_db = if rms_norm > 1e-10 {
            20.0 * rms_norm.log10()
        } else {
            -100.0
        };
        println!(
            "Normalized: RMS={:.6} ({:.1} dBFS), Peak={:.6}",
            rms_norm, rms_norm_db, peak_norm
        );

        audio_norm
    };

    // Apply output gain
    let mut audio_final = audio_final;
    if args.output_gain != 1.0 {
        for s in audio_final.iter_mut() {
            *s *= args.output_gain;
        }
        let gain_rms = compute_rms(&audio_final);
        let gain_peak = audio_final.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        let gain_rms_db = if gain_rms > 1e-10 {
            20.0 * gain_rms.log10()
        } else {
            -100.0
        };
        println!(
            "Output gain: ×{:.2} → RMS={:.6} ({:.1} dBFS), Peak={:.6}",
            args.output_gain, gain_rms, gain_rms_db, gain_peak
        );
    }

    // Verify peak (allow >1.0 for float32 output)
    let final_peak = audio_final.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if !args.float32_out && final_peak > 1.0 {
        return Err(format!("ERROR: Peak {:.6} exceeds 1.0 for int16 output! Use --float32-out or reduce --output-gain", final_peak).into());
    }

    println!("\nWriting: {:?}", args.output);
    write_wav(
        &args.output,
        &audio_final,
        args.sample_rate,
        args.float32_out,
    )?;

    let written_rms = compute_rms(&audio_final);
    let written_std = compute_std(&audio_final);
    let written_rms_db = if written_rms > 1e-10 {
        20.0 * written_rms.log10()
    } else {
        -100.0
    };
    println!(
        "Written: RMS={:.6} ({:.1} dBFS), Peak={:.6}, Std={:.6}, Duration={:.2}s",
        written_rms,
        written_rms_db,
        final_peak,
        written_std,
        audio_final.len() as f64 / args.sample_rate as f64
    );

    if !args.no_normalize && written_std < 0.01 {
        println!(
            "⚠ WARNING: Low variance (std={:.6}), audio may be crushed",
            written_std
        );
    }

    println!("Done!");
    Ok(())
}

fn pixel_to_magnitude(
    norm_value: f64,
    mapping: MappingType,
    gamma: f64,
    scale: f64,
    db_min: f64,
    db_max: f64,
    auto_norm: Option<(f64, f64)>,
) -> f64 {
    let mut norm = norm_value.clamp(0.0, 1.0);
    if let Some((low, high)) = auto_norm {
        if high > low + f64::EPSILON {
            norm = ((norm - low) / (high - low)).clamp(0.0, 1.0);
        } else {
            norm = 0.0;
        }
    }

    match mapping {
        MappingType::Linear => norm * scale,
        MappingType::Power => norm.powf(gamma) * scale,
        MappingType::Db => {
            if norm <= 0.0 {
                0.0
            } else {
                let db = norm * (db_max - db_min) + db_min;
                10f64.powf(db / 20.0)
            }
        }
    }
}

fn sample_bilinear(img: &ImageBuffer<Luma<f32>, Vec<f32>>, x: f64, y: f64) -> f64 {
    let width = img.width();
    let height = img.height();
    if width == 0 || height == 0 {
        return 0.0;
    }

    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);

    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(max_x);
    let y1 = (y0 + 1.0).min(max_y);

    let tx = (x - x0) as f32;
    let ty = (y - y0) as f32;

    let p00 = img.get_pixel(x0 as u32, y0 as u32).0[0];
    let p10 = img.get_pixel(x1 as u32, y0 as u32).0[0];
    let p01 = img.get_pixel(x0 as u32, y1 as u32).0[0];
    let p11 = img.get_pixel(x1 as u32, y1 as u32).0[0];

    let a = p00 * (1.0 - tx) + p10 * tx;
    let b = p01 * (1.0 - tx) + p11 * tx;
    (a * (1.0 - ty) + b * ty) as f64
}

fn per_frame_normalize(mag: &mut [Vec<f64>], floor: f64) {
    let num_frames = mag[0].len();
    let num_bins = mag.len();
    for t in 0..num_frames {
        let sum: f64 = (0..num_bins).map(|f| mag[f][t]).sum();
        let mean = sum / num_bins as f64;
        if mean > floor {
            let gain = (1.0 / mean).min(10.0);
            for f in 0..num_bins {
                mag[f][t] *= gain;
            }
        }
    }
}

fn percentile_clamp(mag: &mut [Vec<f64>], p_low: f64, p_high: f64) -> (f64, f64) {
    let mut vals: Vec<f64> = mag.iter().flat_map(|r| r.iter().copied()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    let idx_low = ((p_low / 100.0) * n as f64) as usize;
    let idx_high = ((p_high / 100.0) * n as f64).min(n as f64 - 1.0) as usize;
    let low_val = vals[idx_low];
    let high_val = vals[idx_high];

    for row in mag.iter_mut() {
        for v in row.iter_mut() {
            *v = v.clamp(low_val, high_val);
        }
    }
    (low_val, high_val)
}

fn smooth_along_freq(mag: &mut [Vec<f64>], kernel: usize) {
    if kernel < 3 || kernel % 2 == 0 {
        return;
    }
    let num_bins = mag.len();
    let num_frames = mag[0].len();
    let half = kernel / 2;
    let mut temp = vec![0.0; num_bins];

    for t in 0..num_frames {
        for f in 0..num_bins {
            let mut sum = 0.0;
            let mut count = 0;
            for k in 0..kernel {
                let idx = f as isize + k as isize - half as isize;
                if idx >= 0 && idx < num_bins as isize {
                    sum += mag[idx as usize][t];
                    count += 1;
                }
            }
            temp[f] = sum / count as f64;
        }
        for f in 0..num_bins {
            mag[f][t] = temp[f];
        }
    }
}

fn smooth_along_time(mag: &mut [Vec<f64>], kernel: usize) {
    if kernel < 3 || kernel % 2 == 0 {
        return;
    }
    let num_bins = mag.len();
    let num_frames = mag[0].len();
    let half = kernel / 2;
    let mut temp = vec![0.0; num_frames];

    for f in 0..num_bins {
        for t in 0..num_frames {
            let mut sum = 0.0;
            let mut count = 0;
            for k in 0..kernel {
                let idx = t as isize + k as isize - half as isize;
                if idx >= 0 && idx < num_frames as isize {
                    sum += mag[f][idx as usize];
                    count += 1;
                }
            }
            temp[t] = sum / count as f64;
        }
        for t in 0..num_frames {
            mag[f][t] = temp[t];
        }
    }
}

fn median_along_time(mag: &mut [Vec<f64>], kernel: usize) {
    if kernel < 3 {
        return;
    }
    let num_bins = mag.len();
    let num_frames = mag[0].len();
    let half = kernel / 2;
    let mut temp = vec![0.0; num_frames];

    for f in 0..num_bins {
        for t in 0..num_frames {
            let mut window = Vec::new();
            for k in 0..kernel {
                let idx = t as isize + k as isize - half as isize;
                if idx >= 0 && idx < num_frames as isize {
                    window.push(mag[f][idx as usize]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            temp[t] = window[window.len() / 2];
        }
        for t in 0..num_frames {
            mag[f][t] = temp[t];
        }
    }
}

fn apply_bandpass(mag: &mut [Vec<f64>], low_hz: f64, high_hz: f64, sr: u32, nfft: usize) -> usize {
    let freq_per_bin = sr as f64 / nfft as f64;
    let mut zeroed = 0;

    for (f, row) in mag.iter_mut().enumerate() {
        let freq = f as f64 * freq_per_bin;
        if (low_hz > 0.0 && freq < low_hz) || (high_hz > 0.0 && freq > high_hz) {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            zeroed += 1;
        }
    }
    zeroed
}

fn spectral_gate(mag: &mut [Vec<f64>], percentile: f64, reduction: f64) -> f64 {
    let num_bins = mag.len();
    let num_frames = mag[0].len();
    let mut reduced_count = 0;
    let total = num_bins * num_frames;

    for f in 0..num_bins {
        let mut vals: Vec<f64> = mag[f].iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((percentile / 100.0) * vals.len() as f64) as usize;
        let noise_floor = vals[idx.min(vals.len() - 1)];
        let threshold = noise_floor * 1.25;

        for t in 0..num_frames {
            if mag[f][t] < threshold {
                mag[f][t] *= reduction;
                reduced_count += 1;
            }
        }
    }
    100.0 * reduced_count as f64 / total as f64
}

fn fast_griffin_lim(
    magnitude: &[Vec<f64>],
    fft_size: usize,
    hop_length: usize,
    iterations: usize,
    momentum: f64,
    preemph: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let num_bins = magnitude.len();
    let num_frames = magnitude[0].len();

    // Random init
    let mut rng = rand::thread_rng();
    let mut stft = vec![vec![Complex64::new(0.0, 0.0); num_frames]; num_bins];
    for f in 0..num_bins {
        for t in 0..num_frames {
            let phase = rng.gen::<f64>() * 2.0 * PI;
            stft[f][t] = Complex64::from_polar(magnitude[f][t], phase);
        }
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut stft_temp = vec![vec![Complex64::new(0.0, 0.0); num_frames]; num_bins];
    let mut z_prev = vec![vec![Complex64::new(0.0, 0.0); num_frames]; num_bins];

    for iter in 0..iterations {
        if iter % 10 == 0 {
            println!("  Iteration {}/{}", iter + 1, iterations);
        }

        let mut audio = istft(&stft, fft_size, hop_length, &ifft, false);

        // Pre-emphasis if requested
        if preemph > 0.0 && iter == 0 {
            pre_emphasis(&mut audio, preemph);
        }

        stft_forward_fn(&audio, fft_size, hop_length, &fft, &mut stft_temp);

        // Fast GL momentum update
        for f in 0..num_bins {
            for t in 0..num_frames {
                let phase = stft_temp[f][t].arg();
                let z_k = Complex64::from_polar(magnitude[f][t], phase);
                let update = z_k + Complex64::new(momentum, 0.0) * (z_k - z_prev[f][t]);
                stft[f][t] = update;
                z_prev[f][t] = z_k;
            }
        }
    }

    println!("  Final iSTFT...");
    let audio = istft(&stft, fft_size, hop_length, &ifft, true);
    Ok(audio)
}

fn stft_forward_fn(
    audio: &[f64],
    fft_size: usize,
    hop_length: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    output: &mut [Vec<Complex64>],
) {
    let num_bins = fft_size / 2 + 1;
    let num_frames = if audio.len() >= fft_size {
        (audio.len() - fft_size) / hop_length + 1
    } else {
        0
    };
    let window = hann_window(fft_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];

    for frame_idx in 0..num_frames.min(output[0].len()) {
        let start = frame_idx * hop_length;
        if start + fft_size > audio.len() {
            break;
        }
        for i in 0..fft_size {
            buffer[i] = Complex64::new(audio[start + i] * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..num_bins {
            output[bin][frame_idx] = buffer[bin];
        }
    }
}

fn istft(
    stft: &[Vec<Complex64>],
    fft_size: usize,
    hop_length: usize,
    ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    print_cola: bool,
) -> Vec<f64> {
    let num_frames = stft[0].len();
    let audio_len = (num_frames - 1) * hop_length + fft_size;
    let mut audio = vec![0.0f64; audio_len];
    let mut window_sum = vec![0.0f64; audio_len];
    let window = hann_window(fft_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];

    for frame_idx in 0..num_frames {
        buffer[0] = stft[0][frame_idx];
        for bin in 1..(fft_size / 2) {
            buffer[bin] = stft[bin][frame_idx];
            buffer[fft_size - bin] = stft[bin][frame_idx].conj();
        }
        if fft_size % 2 == 0 && stft.len() > fft_size / 2 {
            buffer[fft_size / 2] = stft[fft_size / 2][frame_idx];
        }
        ifft.process(&mut buffer);

        let ifft_scale = 1.0 / fft_size as f64;
        let start = frame_idx * hop_length;
        for i in 0..fft_size {
            if start + i < audio_len {
                audio[start + i] += buffer[i].re * ifft_scale * window[i];
                window_sum[start + i] += window[i] * window[i];
            }
        }
    }

    for i in 0..audio_len {
        if window_sum[i] > 1e-10 {
            audio[i] /= window_sum[i];
        }
    }

    if print_cola {
        let mut min_ws = f64::INFINITY;
        let mut max_ws = 0.0f64;
        for &ws in &window_sum {
            if ws > 1e-10 {
                min_ws = min_ws.min(ws);
                max_ws = max_ws.max(ws);
            }
        }
        println!("  COLA: [{:.6}, {:.6}]", min_ws, max_ws);
    }

    audio
}

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - ((2.0 * PI * i as f64) / (size as f64 - 1.0)).cos()))
        .collect()
}

fn pre_emphasis(audio: &mut [f64], coeff: f64) {
    for i in (1..audio.len()).rev() {
        audio[i] -= coeff * audio[i - 1];
    }
}

fn de_emphasis(audio: &mut [f64], coeff: f64) {
    for i in 1..audio.len() {
        audio[i] += coeff * audio[i - 1];
    }
}

fn dc_block(audio: &mut [f64], sr: u32) {
    let cutoff = 20.0;
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sr as f64;
    let alpha = rc / (rc + dt);
    let mut prev_in = 0.0;
    let mut prev_out = 0.0;
    for sample in audio.iter_mut() {
        let out = alpha * (prev_out + *sample - prev_in);
        prev_in = *sample;
        prev_out = out;
        *sample = out;
    }
}

fn one_pole_lp(audio: &mut [f64], cutoff: f64, sr: u32) {
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sr as f64;
    let alpha = dt / (rc + dt);
    let mut prev = 0.0;
    for sample in audio.iter_mut() {
        prev = prev + alpha * (*sample - prev);
        *sample = prev;
    }
}

fn apply_fades(audio: &mut [f64], fade_in_pct: f64, fade_out_pct: f64) {
    let len = audio.len();
    if len == 0 {
        return;
    }

    // Fade in
    if fade_in_pct > 0.0 {
        let fade_in_samples = ((len as f64 * fade_in_pct).round() as usize).min(len / 2);
        for i in 0..fade_in_samples {
            let gain = i as f64 / fade_in_samples as f64;
            audio[i] *= gain;
        }
    }

    // Fade out
    if fade_out_pct > 0.0 {
        let fade_out_samples = ((len as f64 * fade_out_pct).round() as usize).min(len / 2);
        let start = len - fade_out_samples;
        for i in 0..fade_out_samples {
            let gain = 1.0 - (i as f64 / fade_out_samples as f64);
            audio[start + i] *= gain;
        }
    }
}

fn compute_rms(audio: &[f64]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|&x| x * x).sum::<f64>() / audio.len() as f64).sqrt()
}

fn compute_std(audio: &[f64]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let mean = audio.iter().sum::<f64>() / audio.len() as f64;
    let variance = audio.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / audio.len() as f64;
    variance.sqrt()
}

fn normalize_audio(
    audio: &mut [f64],
    target_rms: f64,
    limiter_ceiling: f64,
    _softclip_knee: f64,
    _softclip_slope: f64,
) {
    if audio.is_empty() {
        return;
    }

    // 1) Compute current stats
    let initial_rms = compute_rms(audio);

    if initial_rms < 1e-12 {
        // emergency tiny boost to avoid NaNs
        for s in audio.iter_mut() {
            *s *= 1000.0;
        }
        let boosted_rms = compute_rms(audio);
        if boosted_rms < 1e-12 {
            return;
        }

        // Apply target gain
        let gain = target_rms / boosted_rms;
        for s in audio.iter_mut() {
            *s *= gain;
        }
    } else {
        // 2) Apply RMS gain to hit target (may exceed ceiling - that's OK!)
        let gain = target_rms / initial_rms;
        println!(
            "  RMS gain: {:.3}× ({:.6} → {:.6})",
            gain, initial_rms, target_rms
        );
        for s in audio.iter_mut() {
            *s *= gain;
        }

        // Check what we have after gain
        let after_gain_rms = compute_rms(audio);
        let after_gain_peak = audio.iter().copied().map(f64::abs).fold(0.0, f64::max);
        println!(
            "  After gain: RMS={:.6}, Peak={:.6}",
            after_gain_rms, after_gain_peak
        );
    }

    // 3) Brick-wall limiter at ceiling (simple hard clip)
    println!("  Limiter ceiling: {:.6}", limiter_ceiling);
    let mut clipped = 0;
    let mut max_found = 0.0;
    let mut first_clip_logged = false;
    for s in audio.iter_mut() {
        let abs_s = s.abs();
        if abs_s > max_found {
            max_found = abs_s;
        }
        if abs_s > limiter_ceiling {
            if !first_clip_logged {
                println!(
                    "  DEBUG: First clip - abs_s={:.6}, ceiling={:.6}, comparison={}",
                    abs_s,
                    limiter_ceiling,
                    abs_s > limiter_ceiling
                );
                first_clip_logged = true;
            }
            *s = s.signum() * limiter_ceiling;
            clipped += 1;
        }
    }
    println!("  Max sample found during limiting: {:.6}", max_found);

    let final_rms = compute_rms(audio);
    let final_peak = audio.iter().copied().map(f64::abs).fold(0.0, f64::max);

    if clipped > 0 {
        let pct = 100.0 * clipped as f64 / audio.len() as f64;
        println!("  Limiter: {} samples ({:.1}%) clipped", clipped, pct);
    } else {
        println!("  Limiter: 0 samples clipped (BUG if peak > ceiling!)");
    }
    println!("  Final: RMS={:.6}, Peak={:.6}", final_rms, final_peak);
}

fn write_wav(
    path: &PathBuf,
    audio: &[f64],
    sample_rate: u32,
    float32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: if float32 { 32 } else { 16 },
        sample_format: if float32 {
            SampleFormat::Float
        } else {
            SampleFormat::Int
        },
    };
    let mut writer = WavWriter::create(path, spec)?;
    if float32 {
        for &s in audio {
            writer.write_sample(s.clamp(-1.0, 1.0) as f32)?;
        }
    } else {
        for &s in audio {
            let q = (s.clamp(-1.0, 1.0) * i16::MAX as f64) as i16;
            writer.write_sample(q)?;
        }
    }
    writer.finalize()?;
    Ok(())
}
