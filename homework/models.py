from pathlib import Path

import torch
import torch.nn as nn

import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]



class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = torch.nn.BatchNorm2d(out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = torch.nn.BatchNorm2d(out_channels)
            self.relu1 = torch.nn.ReLU()
            self.relu2 = torch.nn.ReLU()
            
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()
            
        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            return self.skip(x0) + x

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        n_blocks = 4
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()
        cnn_layers = [
            torch.nn.Conv2d(3, in_channels, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU()
        ]
        
        c1 = in_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 6)

        # return logits
        return self.network(z).mean(dim=-1).mean(dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


# class Detector(nn.Module):
#     def __init__(self,         
#                 in_channels: int = 3,
#                 num_classes: int = 3,):
#         super().__init__()
#         self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
#         self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
#         self.down1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 3 -> 16
#         self.down2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16 -> 32

#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # (B, 32, H/4, W/4)
#         self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # (B, 16, H/2, W/2)

#         self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)



#     def forward(self, x):
#         z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

#         z = torch.relu(self.down1(z))
#         z = torch.relu(self.down2(z))

#         # Upsample
#         z = torch.relu(self.up1(z))
#         logits = self.up2(z)  # Shape: (B, num_classes, 96, 128)

#         # Depth regression
#         depth = self.depth_conv(z)  # Shape: (B, 1, 96, 128)

#         return logits, depth.squeeze(1)
#     def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Used for inference, takes an image and returns class labels and normalized depth.
#         This is what the metrics use as input (this is what the grader will use!).

#         Args:
#             x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

#         Returns:
#             tuple of (torch.LongTensor, torch.FloatTensor):
#                 - pred: class labels {0, 1, 2} with shape (b, h, w)
#                 - depth: normalized depth [0, 1] with shape (b, h, w)
#         """
#         logits, raw_depth = self(x)
#         pred = logits.argmax(dim=1)

#         # Optional additional post-processing for depth only if needed
#         depth = raw_depth

#         return pred, depth

class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        # Output heads
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        d1 = self.down1(z)
        d2 = self.down2(d1)
        u1 = self.up1(d2) + d1  # Skip connection
        u2 = self.up2(u1)
        
        logits = self.segmentation_head(u2)
        raw_depth = torch.sigmoid(self.depth_head(u2))  # Ensure depth is in range [0,1]
        
        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
