import torch

from tadisr.tiling import tiled_predict


def test_tiled_predict_reconstructs_identity_and_mask() -> None:
    image = torch.rand(1, 3, 1021, 887)

    def predict(tile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tile, tile[:, :1]

    sr, mask = tiled_predict(image, predict, tile=384, overlap=128)
    torch.testing.assert_close(sr, image, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mask, image[:, :1], rtol=1e-4, atol=1e-4)
