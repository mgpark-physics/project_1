# Check which PyTorch accelerator is available
import torch

device = "cpu"

curr = torch.accelerator.current_accelerator(
    check_available=True
)

match curr:
    case torch.device():
        device = curr.type
    case None:
        device = "cpu"

print(f"Using {device} device")