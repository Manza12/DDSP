import torch
import math
from scipy.io.wavfile import read as wave_read
from scipy.io.wavfile import write as wave_write
import torch.nn.functional as func


#Fast frequency domain pytorch convolution for long kernel 
def _complex_mul(a, b):
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

def convolve(signal, kernel, mode='full'):
    """ 
    This code is copied from the pytorch github issue #21462
    
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.
    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    :rtype torch.Tensor:
    """
    m = signal.size(-1)
    n = kernel.size(-1)
    if mode == 'full':
        truncate = m + n - 1
    elif mode == 'valid':
        truncate = max(m, n) - min(m, n) + 1
    elif mode == 'same':
        truncate = max(m, n)
    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up to next power of 2 for cheaper fft.
    fast_ftt_size = 2 ** math.ceil(math.log2(padded_size))
    f_signal = torch.rfft(torch.nn.functional.pad(signal, (0, fast_ftt_size - m)), 1, onesided=False)
    f_kernel = torch.rfft(torch.nn.functional.pad(kernel, (0, fast_ftt_size - n)), 1, onesided=False)
    f_result = _complex_mul(f_signal, f_kernel)
    result = torch.irfft(f_result, 1, onesided=False)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx: start_idx + truncate]


def add_reverb(dry_signal, impulse_response):

    ir_length = impulse_response.shape[-1]
    # signal_length = dry_signal.shape[-1]
    # max_length = max(ir_length, signal_length)

    dry_signal_paded = func.pad(dry_signal, [ir_length//2, ir_length//2])
    # impulse_response = func.pad(impulse_response, (0, max_length - ir_length))

    wet_signal = torch.conv1d(dry_signal_paded.unsqueeze(1), impulse_response.unsqueeze(0).unsqueeze(0))
    wet_signal = wet_signal.squeeze(1)

    return 0.1*(0.3 * wet_signal + 0.7 * dry_signal)


if __name__ == "__main__":
    import numpy as np

    rate, l = wave_read('test.wav')
    l = l.astype(float)/max(abs(l))
    l = torch.from_numpy(l)
    l = l.type(torch.float)
    l = l.expand(6, l.shape[-1])

    ir_filename = "ir.wav"
    test_tensor_wet = add_reverb(l, ir_filename)
    wave_write('test_return.wav', 16000, np.array(test_tensor))