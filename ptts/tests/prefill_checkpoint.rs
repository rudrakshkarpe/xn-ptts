//! The cache-only prefill must leave exactly the same state as a full forward.
#![cfg(feature = "hf")]

use ptts::transformer::{LayerAttentionState, StreamingTransformerState};
use ptts::tts_model::{TTSConfig, TTSModel};
use xn::{CpuDevice, Tensor};

type Q = xn::quantized::Q80F32;

fn same_state(
    a: &StreamingTransformerState<f32, CpuDevice>,
    b: &StreamingTransformerState<f32, CpuDevice>,
) {
    assert_eq!(a.layer_states.len(), b.layer_states.len());
    for (index, (a, b)) in a.layer_states.iter().zip(&b.layer_states).enumerate() {
        let (LayerAttentionState::FlowLm(a), LayerAttentionState::FlowLm(b)) = (a, b) else {
            panic!("expected flow transformer");
        };
        assert_eq!(a.current_end, b.current_end);
        assert_eq!(a.k_cache.to_vec().unwrap(), b.k_cache.to_vec().unwrap(), "layer {index} K");
        assert_eq!(a.v_cache.to_vec().unwrap(), b.v_cache.to_vec().unwrap(), "layer {index} V");
    }
}

#[test]
#[ignore = "requires PTTS_TEST_MODEL with local Phonon q8 weights; no download"]
fn cache_only_prefill_preserves_every_layer_and_subsequent_decode_exactly() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::var("PTTS_TEST_MODEL")?);
    let cfg: TTSConfig = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let vb = ptts::loader::load_weights::<Q>(&path.join("model.q8.gguf"), &CpuDevice)?;
    let tokenizer = ptts::tok::Tok::open(&path.join("tokenizer.json"))?;
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg, &Default::default())?;
    let transformer = &model.flow_lm.transformer;
    let width = cfg.flow_lm.d_model;
    let mut reference = transformer.init_state(1, 128)?;
    let mut optimized = transformer.init_state(1, 128)?;
    // Successive prefixes test both an empty cache and nonzero position/RoPE offsets.
    for (round, length) in [1, 7, 31].into_iter().enumerate() {
        let values = (0..length * width).map(|i| ((i + round * 37) as f32 * 0.017).sin()).collect();
        let input = Tensor::from_vec(values, (1, length, width), &CpuDevice)?;
        transformer.forward(&input, &mut reference)?;
        transformer.prefill(&input, &mut optimized)?;
        same_state(&reference, &optimized);
        for step in 0..3 {
            let input = Tensor::from_vec(
                (0..width).map(|i| ((i + step) as f32 * 0.03).cos()).collect(),
                (1, 1, width),
                &CpuDevice,
            )?;
            let a = transformer.forward(&input, &mut reference)?.to_vec()?;
            let b = transformer.forward(&input, &mut optimized)?.to_vec()?;
            assert_eq!(a, b, "round {round}, decode {step}");
            same_state(&reference, &optimized);
        }
    }
    Ok(())
}
