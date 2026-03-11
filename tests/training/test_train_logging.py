from lerobot.scripts.lerobot_train import (
    _format_supervision_batch_log,
    _merge_microbatch_output_dicts,
)


def test_merge_microbatch_output_dicts_keeps_exact_batch_supervision_fraction():
    merged = _merge_microbatch_output_dicts(
        [
            {
                "loss": 1.0,
                "action_supervised_fraction": 0.5,
                "batch_action_supervised_samples": 4.0,
                "_action_supervised_denominator": 8.0,
                "latent_supervised_fraction": 0.5,
                "batch_latent_supervised_samples": 4.0,
                "_latent_supervised_denominator": 8.0,
            },
            {
                "loss": 3.0,
                "action_supervised_fraction": 0.25,
                "batch_action_supervised_samples": 6.0,
                "_action_supervised_denominator": 24.0,
                "latent_supervised_fraction": 1.0,
                "batch_latent_supervised_samples": 24.0,
                "_latent_supervised_denominator": 24.0,
            },
        ]
    )

    assert merged["loss"] == 2.0
    assert merged["action_supervised_fraction"] == 0.375
    assert merged["batch_action_supervised_samples"] == 10.0
    assert merged["batch_action_supervised_denominator"] == 32.0
    assert merged["batch_action_supervised_fraction"] == 0.3125
    assert merged["latent_supervised_fraction"] == 0.75
    assert merged["batch_latent_supervised_samples"] == 28.0
    assert merged["batch_latent_supervised_denominator"] == 32.0
    assert merged["batch_latent_supervised_fraction"] == 0.875


def test_format_supervision_batch_log_includes_counts_and_fractions():
    line = _format_supervision_batch_log(
        {
            "batch_action_supervised_fraction": 0.3125,
            "batch_action_supervised_samples": 10.0,
            "batch_action_supervised_denominator": 32.0,
            "batch_latent_supervised_fraction": 0.875,
            "batch_latent_supervised_samples": 28.0,
            "batch_latent_supervised_denominator": 32.0,
        }
    )

    assert line == "batch_sup action=0.312 (10.0/32.0) latent=0.875 (28.0/32.0)"
