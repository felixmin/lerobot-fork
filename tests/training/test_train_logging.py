from lerobot.scripts.lerobot_train import (
    _build_exact_scaled_loss,
    _format_supervision_batch_log,
    _merge_microbatch_output_dicts,
)
import torch


def test_merge_microbatch_output_dicts_keeps_exact_batch_supervision_fraction():
    merged = _merge_microbatch_output_dicts(
        [
            {
                "loss": 1.0,
                "action_loss": 1.0,
                "_action_loss_denominator_exact": 8.0,
                "action_supervised_fraction": 0.5,
                "batch_action_supervised_samples": 4.0,
                "_action_supervised_denominator": 8.0,
                "latent_loss": 4.0,
                "_latent_loss_denominator_exact": 8.0,
                "latent_supervised_fraction": 0.5,
                "batch_latent_supervised_samples": 4.0,
                "_latent_supervised_denominator": 8.0,
            },
            {
                "loss": 3.0,
                "action_loss": 5.0,
                "_action_loss_denominator_exact": 24.0,
                "action_supervised_fraction": 0.25,
                "batch_action_supervised_samples": 6.0,
                "_action_supervised_denominator": 24.0,
                "latent_loss": 2.0,
                "_latent_loss_denominator_exact": 24.0,
                "latent_supervised_fraction": 1.0,
                "batch_latent_supervised_samples": 24.0,
                "_latent_supervised_denominator": 24.0,
            },
        ]
    )

    assert merged["loss"] == 2.0
    assert merged["action_loss"] == 4.0
    assert merged["action_supervised_fraction"] == 0.375
    assert merged["batch_action_supervised_samples"] == 10.0
    assert merged["batch_action_supervised_denominator"] == 32.0
    assert merged["batch_action_supervised_fraction"] == 0.3125
    assert merged["latent_loss"] == 2.5
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


def test_build_exact_scaled_loss_uses_global_denominator_weights():
    output_dict = {
        "_action_loss_tensor": torch.tensor(2.0),
        "_action_loss_denominator_exact": 8.0,
        "_latent_loss_tensor": torch.tensor(4.0),
        "_latent_loss_denominator_exact": 16.0,
    }

    scaled = _build_exact_scaled_loss(
        output_dict=output_dict,
        action_total=32.0,
        latent_total=32.0,
        action_weight=1.5,
        latent_weight=0.5,
    )

    assert torch.isclose(scaled, torch.tensor(1.75))
