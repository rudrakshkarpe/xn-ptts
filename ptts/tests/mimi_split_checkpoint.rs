//! Opt-in streaming-equivalence checks with local weights; no downloads.
#![cfg(feature = "hf")]
use ptts::tts_model::{TTSConfig, TTSModel};
use xn::{CpuDevice, Tensor};
type Q = xn::quantized::Q80F32;

fn model() -> anyhow::Result<TTSModel<Q>> {
    let root = std::path::PathBuf::from(std::env::var("PTTS_TEST_MODEL")?);
    let config: TTSConfig = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let weights = ptts::loader::load_weights::<Q>(&root.join("model.q8.gguf"), &CpuDevice)?;
    let tok = ptts::tok::Tok::open(&root.join("tokenizer.json"))?;
    Ok(TTSModel::load(&weights, Box::new(tok), &config, &Default::default())?)
}

#[test]
#[ignore = "requires PTTS_TEST_MODEL; also run with PTTS_EXPERIMENT_PACKED_MIMI=1"]
fn splitting_first_frame_preserves_pcm_and_later_streaming_state() -> anyhow::Result<()> {
    let model = model()?;
    let dim = model.flow_lm.ldim;
    let latent = |frames, phase| {
        Tensor::from_vec(
            (0..frames * dim).map(|i| ((i + phase) as f32 * 0.17).sin() * 0.2).collect(),
            (1, frames, dim),
            &CpuDevice,
        )
    };
    let first = latent(1, 0)?;
    let later = latent(4, 53)?;
    let mut reference = model.init_mimi_state(1)?;
    let mut expected = model.decode_latent(&first, &mut reference)?.to_vec()?;
    let first_len = expected.len();
    expected.extend(model.decode_latent(&later, &mut reference)?.to_vec()?);
    for waveform_only in [false, true] {
        for steps in [0, 1, 4, 8, 16, 32] {
            let mut state = model.init_mimi_state(1)?;
            let mut actual = Vec::new();
            let mut chunks = Vec::new();
            assert!(model.decode_latent_split(
                &first,
                &mut state,
                steps,
                waveform_only,
                |audio| {
                    let pcm = audio.to_vec()?;
                    chunks.push(pcm.len());
                    actual.extend(pcm);
                    Ok(true)
                }
            )?);
            assert_eq!(actual.len(), first_len);
            if steps == 4 {
                assert_eq!(chunks, vec![480, 1440]);
            }
            actual.extend(model.decode_latent(&later, &mut state)?.to_vec()?);
            assert_eq!(expected.len(), actual.len());
            assert!(actual.iter().all(|v| v.is_finite()));
            let maximum =
                expected.iter().zip(&actual).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(maximum < 1e-5, "steps={steps} waveform_only={waveform_only} error={maximum}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires PTTS_TEST_MODEL"]
fn cancelling_after_the_first_piece_does_not_emit_the_remainder() -> anyhow::Result<()> {
    let model = model()?;
    let latent = Tensor::zeros((1, 1, model.flow_lm.ldim), &CpuDevice)?;
    for waveform_only in [false, true] {
        let mut state = model.init_mimi_state(1)?;
        let mut count = 0;
        assert!(!model.decode_latent_split(&latent, &mut state, 4, waveform_only, |audio| {
            count += 1;
            assert_eq!(audio.elem_count(), 480);
            Ok(false)
        })?);
        assert_eq!(count, 1);
    }
    Ok(())
}
