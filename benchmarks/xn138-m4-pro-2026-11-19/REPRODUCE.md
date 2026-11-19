# Reproduction

Use an Apple Silicon machine with SME2 support and the same Rust toolchain and build flags for all variants. This run used prepared release binaries built on November 17; their hashes are in `manifest.json`. The observed host toolchain is rustc 1.92.0. All three variants use the identical `reproduction/bench.rs` harness.

Create three checkouts of https://github.com/gradium-ai/xn-ptts:

| Variant | ptts revision | Patch to apply |
|---|---|---|
| baseline | e5110afd3ead725c940b6411eec4985fb8b111aa | reproduction/baseline.patch |
| xn138 | e5110afd3ead725c940b6411eec4985fb8b111aa | reproduction/xn138.patch |
| pr322 | 70d748173cfe0250a9e116e9706eb96809e1630b | None; harness is committed |

Apply each patch with `git apply` from its checkout. The xn138 patch pins xn to `7e9ffab9b82c7a45a953140a4f2b2578cf356b0a`. The baseline uses released xn 0.2.13; PR #322 pins its proposed owned-operator API. Keep each checkout's Cargo.lock.

Build each variant twice, copying each binary before the next build:

```sh
cargo build --locked --release -p ptts --example bench --no-default-features --features hf,accelerate,kai
cargo build --locked --release -p ptts --example bench --no-default-features --features hf,kai
```

The repository's `.cargo/config.toml` enables `-C target-cpu=native`. Place the six binaries in an empty results directory under `binaries/`, named `ptts-{baseline,xn138,pr322}-{acckai,kai}`. The model directory must contain `model.q8.gguf`, `config.json`, `tokenizer.json`, and `default-voice.safetensors`; verify the hashes in the manifest.

```sh
python3 run.py --root /path/to/fresh-results --model /path/to/phonon-31m-en --mode timing
python3 run.py --root /path/to/fresh-results --model /path/to/phonon-31m-en --mode parity
```

Run the modes sequentially. Timing records all chunk arrival times without capturing PCM; parity captures PCM separately. An existing completed result is reused, so use a new results directory for an independent trial matrix. For a controlled repeat, use AC power and an idle desktop and record power, thermal state, and background load. Do not combine a repeat under different conditions with these trials.

The parity mode additionally runs the xn138 binary with `GEMM_PACKED_LHS_CACHE_MB=0` and compares its output with that binary's cache-enabled output. This ablation retains the rest of #138's implementation; it is not a replacement for the released baseline.
