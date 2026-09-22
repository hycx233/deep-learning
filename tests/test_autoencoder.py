import unittest

import numpy as np
import torch

from src.autoencoder import ConvAutoencoder, masked_mse, upper_triangle_mask


class AutoencoderTests(unittest.TestCase):
    def test_autoencoder_shapes(self) -> None:
        model = ConvAutoencoder(latent_dim=12)
        inputs = torch.rand(3, 1, 128, 128)
        latent = model.encode(inputs)
        reconstruction = model(inputs)

        self.assertEqual(tuple(latent.shape), (3, 12))
        self.assertEqual(tuple(reconstruction.shape), tuple(inputs.shape))

    def test_masked_mse_uses_only_selected_triangle(self) -> None:
        target = torch.zeros(1, 1, 4, 4)
        prediction = target.clone()
        prediction[0, 0, 0, 3] = 2.0
        prediction[0, 0, 3, 0] = 100.0
        mask = upper_triangle_mask(4, diagonal_exclusion=0)

        error = masked_mse(prediction, target, mask, reduction="none")

        self.assertTrue(np.isclose(float(error[0]), 4.0 / 6.0))

    def test_masked_mse_ignores_invalid_pixels_per_window(self) -> None:
        target = torch.zeros(2, 1, 4, 4)
        prediction = target.clone()
        prediction[1, 0, 0, 3] = 1.0
        sample_mask = torch.ones(2, 4, 4, dtype=torch.bool)
        sample_mask[1, 0, 3] = False
        loss_mask = sample_mask & upper_triangle_mask(4, diagonal_exclusion=0)

        error = masked_mse(prediction, target, loss_mask, reduction="none")

        self.assertEqual(error.tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
