# Examples

`input.jpg` is a small sample spectrogram-style image you can use for quick tests:

```bash
cargo run --release -- -i examples/input.jpg -o output.wav
```

Add your own assets to this directory as needed; `.gitignore` prevents accidental commits of large generated WAV files.
