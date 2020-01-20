import torch
import torch.nn.functional as func


def add_reverb(dry_signal, impulse_response):
    ir_length = impulse_response.shape[-1]
    dry_signal_paded = func.pad(dry_signal, [ir_length//2, ir_length//2])
    wet_signal = torch.conv1d(dry_signal_paded.unsqueeze(1),
                              impulse_response.unsqueeze(0).unsqueeze(0))
    wet_signal = wet_signal.squeeze(1)

    return 0.3 * wet_signal + 0.7 * dry_signal
