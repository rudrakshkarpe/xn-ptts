//! Research microbenchmark: identical f32 GEMMs with reused output buffers.
//! Compares Accelerate with KleidiAI SME2 including vs excluding repeated weight packing.
//! This does not change the serving path or claim an end-to-end improvement.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod probe {
    use std::{ffi::c_void, hint::black_box, time::Instant};
    // Signatures match xn 0.2.13's kai_f32.rs and its vendored KleidiAI headers.
    unsafe extern "C" {
        fn kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
        fn kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
        fn kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
        fn kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
        fn kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(
            m: usize,
            k: usize,
            mr: usize,
            kr: usize,
            sr: usize,
        ) -> usize;
        fn kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
            n: usize,
            k: usize,
        ) -> usize;
        fn kai_run_lhs_pack_f32p2vlx1_f32_sme(
            m: usize,
            k: usize,
            mr: usize,
            kr: usize,
            sr: usize,
            start: usize,
            x: *const c_void,
            stride: usize,
            packed: *mut c_void,
        );
        fn kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
            groups: usize,
            n: usize,
            k: usize,
            nr: usize,
            kr: usize,
            sr: usize,
            stride: usize,
            w: *const c_void,
            bias: *const c_void,
            scale: *const c_void,
            packed: *mut c_void,
            extra: usize,
            params: *const c_void,
        );
        fn kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
            m: usize,
            n: usize,
            k: usize,
            x: *const c_void,
            w: *const c_void,
            out: *mut c_void,
            row_stride: usize,
            col_stride: usize,
            lo: f32,
            hi: f32,
        );
        fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn sgemm_(
            ta: *const i8,
            tb: *const i8,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f32,
            a: *const f32,
            lda: *const i32,
            b: *const f32,
            ldb: *const i32,
            beta: *const f32,
            c: *mut f32,
            ldc: *const i32,
        );
    }

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / 16777216. - 0.5) * 0.2
            })
            .collect()
    }

    pub fn run() {
        xn::set_num_threads(4);
        assert!(xn::quantized::kai::active(), "requires an SME2 CPU with KleidiAI enabled");
        let mut observations = Vec::new();
        for (label, n, k) in [
            ("mimi_ffn_expand", 2048, 512),
            ("mimi_ffn_contract", 512, 2048),
            ("mimi_qkv", 1536, 512),
        ] {
            for m in [4usize, 16] {
                let x = data(m * k, 42);
                let w = data(n * k, 96);
                let bias = vec![0f32; n];
                let mut reference = vec![0f32; m * n];
                let mut output = vec![0f32; m * n];
                // SAFETY: CPU support was checked above. Buffer sizes are queried from the
                // matching packer, all dimensions/strides describe the allocated dense arrays,
                // and every pointer remains valid for the synchronous calls below.
                unsafe {
                    let mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                    let nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                    let kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                    let sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                    let xs = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
                    let ws =
                        kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k);
                    let mut xp = vec![0f32; xs.div_ceil(4)];
                    let mut wp = vec![0f32; ws.div_ceil(4)];
                    let pack_weights = |buffer: &mut Vec<f32>| {
                        kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
                            1,
                            n,
                            k,
                            nr,
                            kr,
                            sr,
                            k * 4,
                            w.as_ptr().cast(),
                            bias.as_ptr().cast(),
                            std::ptr::null(),
                            buffer.as_mut_ptr().cast(),
                            0,
                            std::ptr::null(),
                        )
                    };
                    pack_weights(&mut wp);
                    cblas_sgemm(
                        101,
                        111,
                        112,
                        m as i32,
                        n as i32,
                        k as i32,
                        1.,
                        x.as_ptr(),
                        k as i32,
                        w.as_ptr(),
                        k as i32,
                        0.,
                        reference.as_mut_ptr(),
                        n as i32,
                    );
                    for trial in 0..5 {
                        let order = if trial % 2 == 0 {
                            ["accelerate", "accelerate_fortran", "sme2_repack", "sme2_prepacked"]
                        } else {
                            ["sme2_prepacked", "sme2_repack", "accelerate_fortran", "accelerate"]
                        };
                        for variant in order {
                            let mut invoke = || {
                                if variant == "accelerate" {
                                    cblas_sgemm(
                                        101,
                                        111,
                                        112,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        1.,
                                        x.as_ptr(),
                                        k as i32,
                                        w.as_ptr(),
                                        k as i32,
                                        0.,
                                        output.as_mut_ptr(),
                                        n as i32,
                                    );
                                } else if variant == "accelerate_fortran" {
                                    // xn's row-major GEMM maps to column-major W^T * X.
                                    // Keep this ABI control separate from the CBLAS result.
                                    sgemm_(
                                        &84,
                                        &78,
                                        &(n as i32),
                                        &(m as i32),
                                        &(k as i32),
                                        &1.,
                                        w.as_ptr(),
                                        &(k as i32),
                                        x.as_ptr(),
                                        &(k as i32),
                                        &0.,
                                        output.as_mut_ptr(),
                                        &(n as i32),
                                    );
                                } else {
                                    if variant == "sme2_repack" {
                                        pack_weights(&mut wp);
                                    }
                                    kai_run_lhs_pack_f32p2vlx1_f32_sme(
                                        m,
                                        k,
                                        mr,
                                        kr,
                                        sr,
                                        0,
                                        x.as_ptr().cast(),
                                        k * 4,
                                        xp.as_mut_ptr().cast(),
                                    );
                                    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                                        m,
                                        n,
                                        k,
                                        xp.as_ptr().cast(),
                                        wp.as_ptr().cast(),
                                        output.as_mut_ptr().cast(),
                                        n * 4,
                                        4,
                                        f32::NAN,
                                        f32::NAN,
                                    );
                                }
                                black_box(&output);
                            };
                            for _ in 0..20 {
                                invoke();
                            }
                            let start = Instant::now();
                            for _ in 0..200 {
                                invoke();
                            }
                            let us = start.elapsed().as_secs_f64() * 1e6 / 200.;
                            let max_error = reference
                                .iter()
                                .zip(&output)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            assert!(
                                output.iter().all(|x| x.is_finite()) && max_error < 1e-4,
                                "kernel mismatch {max_error}"
                            );
                            observations.push(serde_json::json!({"shape":label,"m":m,"n":n,"k":k,"trial":trial,"variant":variant,"mean_us":us,"max_abs_error":max_error,"packed_weight_bytes":ws}));
                        }
                    }
                }
            }
        }
        println!("{}", serde_json::to_string_pretty(&observations).unwrap());
    }
}

fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    probe::run();
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    eprintln!("This research probe requires Apple Silicon with SME2.");
}
