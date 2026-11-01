//! Opt-in research prototype: model-owned packed f32 weights for Mimi.
//! Original weights remain available for the baseline, so packing adds memory.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kai"))]
mod native {
    use std::ffi::c_void;
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

    }

    pub(crate) struct PackedF32 {
        weights: Vec<f32>,
        n: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    }
    impl PackedF32 {
        pub(crate) fn load<Q: xn::BackendQ>(
            vb: &xn::nn::var_builder::Path<Q::B>,
            n: usize,
            k: usize,
        ) -> xn::Result<Option<Box<Self>>> {
            if std::env::var("PTTS_EXPERIMENT_PACKED_MIMI").as_deref() != Ok("1")
                || std::any::TypeId::of::<Q>()
                    != std::any::TypeId::of::<xn::Unquantized<f32, xn::CpuDevice>>()
                || !xn::quantized::kai::active()
            {
                return Ok(None);
            }
            let w: xn::Tensor<f32, Q::B> = vb.tensor("weight", (n, k))?;
            let w = w.to_vec()?;
            Ok(Some(Box::new(Self::pack(&w, n, k)?)))
        }

        fn pack(w: &[f32], n: usize, k: usize) -> xn::Result<Self> {
            if n == 0
                || k == 0
                || n.checked_mul(k) != Some(w.len())
                || !xn::quantized::kai::active()
            {
                xn::bail!("invalid packed f32 weights or unsupported CPU");
            }
            let bias = vec![0f32; n];
            // SAFETY: the runtime CPU check above guarantees SME2. Dimensions match the
            // source allocation; the packer's own size query bounds its output buffer.
            unsafe {
                let mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                let nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                let kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                let sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                let bytes = kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k);
                let mut weights = vec![0f32; bytes.div_ceil(4)];
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
                    weights.as_mut_ptr().cast(),
                    0,
                    std::ptr::null(),
                );
                eprintln!(
                    "experimental packed Mimi matrix: n={n} k={k} bytes={}",
                    weights.len() * 4
                );
                Ok(Self { weights, n, k, mr, kr, sr })
            }
        }
        #[tracing::instrument(level="trace",target="ptts_profile",name="packed_mimi_linear",skip_all,fields(n=self.n,k=self.k))]
        pub(crate) fn forward<T: xn::WithDTypeF, B: xn::Backend>(
            &self,
            x: &xn::Tensor<T, B>,
        ) -> xn::Result<xn::Tensor<T, B>> {
            if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>()
                || std::any::TypeId::of::<B>() != std::any::TypeId::of::<xn::CpuDevice>()
                || x.dims().last().copied() != Some(self.k)
            {
                xn::bail!("incompatible packed f32 input");
            }
            let input: Vec<f32> = x.to_vec()?.into_iter().map(|v| v.to_f32()).collect();
            let m = input.len() / self.k;
            let mut out = vec![0f32; m * self.n];
            if m != 0 {
                // SAFETY: only validated CPU f32 tensors reach here. Input/output sizes are
                // derived from their dimensions; immutable packed weights have model lifetime.
                // Scratch is request-local, so concurrent calls share no mutable buffers.
                unsafe {
                    let bytes = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(
                        m, self.k, self.mr, self.kr, self.sr,
                    );
                    let mut scratch = vec![0f32; bytes.div_ceil(4)];
                    kai_run_lhs_pack_f32p2vlx1_f32_sme(
                        m,
                        self.k,
                        self.mr,
                        self.kr,
                        self.sr,
                        0,
                        input.as_ptr().cast(),
                        self.k * 4,
                        scratch.as_mut_ptr().cast(),
                    );
                    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                        m,
                        self.n,
                        self.k,
                        scratch.as_ptr().cast(),
                        self.weights.as_ptr().cast(),
                        out.as_mut_ptr().cast(),
                        self.n * 4,
                        4,
                        f32::NAN,
                        f32::NAN,
                    );
                }
            }
            let mut shape = x.dims().to_vec();
            *shape.last_mut().expect("validated input rank") = self.n;
            xn::Tensor::from_vec(out.into_iter().map(T::from_f32).collect(), shape, x.device())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::PackedF32;
        use xn::{CpuDevice, Tensor};

        #[test]
        fn packed_tail_shapes_and_concurrent_calls_match_scalar_reference() -> xn::Result<()> {
            if !xn::quantized::kai::active() {
                return Ok(());
            }
            // Odd dimensions exercise packer padding; rank-three inputs exercise flattening.
            let (n, k) = (19, 37);
            let weights: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
            let packed = PackedF32::pack(&weights, n, k)?;
            std::thread::scope(|scope| {
                let handles: Vec<_> = [1, 2, 3, 8, 17]
                    .into_iter()
                    .map(|rows| {
                        let packed = &packed;
                        let weights = &weights;
                        scope.spawn(move || -> xn::Result<()> {
                            let input: Vec<f32> =
                                (0..rows * k).map(|i| (i as f32 * 0.07).cos()).collect();
                            let tensor = Tensor::from_vec(input.clone(), (1, rows, k), &CpuDevice)?;
                            let actual = packed.forward(&tensor)?;
                            assert_eq!(actual.dims(), &[1, rows, n]);
                            for (index, value) in actual.to_vec()?.iter().enumerate() {
                                let (r, c) = (index / n, index % n);
                                let expected: f32 =
                                    (0..k).map(|j| input[r * k + j] * weights[c * k + j]).sum();
                                assert!(
                                    value.is_finite() && (value - expected).abs() < 1e-5,
                                    "row={r} col={c}"
                                );
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap()?;
                }
                xn::Result::Ok(())
            })?;
            let empty = Tensor::<f32, _>::zeros((1, 0, k), &CpuDevice)?;
            assert_eq!(packed.forward(&empty)?.dims(), &[1, 0, n]);
            assert!(packed.forward(&Tensor::<f32, _>::zeros((2, k + 1), &CpuDevice)?).is_err());
            assert!(PackedF32::pack(&[], 0, k).is_err());
            Ok(())
        }
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kai"))]
pub(crate) use native::PackedF32;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "kai")))]
pub(crate) struct PackedF32;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "kai")))]
impl PackedF32 {
    pub(crate) fn load<Q: xn::BackendQ>(
        _: &xn::nn::var_builder::Path<Q::B>,
        _: usize,
        _: usize,
    ) -> xn::Result<Option<Box<Self>>> {
        Ok(None)
    }
    pub(crate) fn forward<T: xn::WithDTypeF, B: xn::Backend>(
        &self,
        _: &xn::Tensor<T, B>,
    ) -> xn::Result<xn::Tensor<T, B>> {
        xn::bail!("SME2 experiment is unavailable")
    }
}
