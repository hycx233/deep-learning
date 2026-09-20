import numpy as np
import torch

from src.autoencoder import ConvAutoencoder, masked_mse, upper_triangle_mask


def test_autoencoder_shapes() -> None:
    model = ConvAutoencoder(latent_dim=12)
    inputs = torch.rand(3, 1, 128, 128)
    latent = model.encode(inputs)
    reconstruction = model(inputs)

    assert latent.shape == (3, 12)
    assert reconstruction.shape == inputs.shape


def test_masked_mse_uses_only_selected_triangle() -> None:
    target = torch.zeros(1, 1, 4, 4)
    prediction = target.clone()
    prediction[0, 0, 0, 3] = 2.0
    prediction[0, 0, 3, 0] = 100.0
    mask = upper_triangle_mask(4, diagonal_exclusion=0)

    error = masked_mse(prediction, target, mask, reduction="none")

    assert np.isclose(float(error[0]), 4.0 / 6.0)
