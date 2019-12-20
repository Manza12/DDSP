from Parameters import *
from DataLoader import read_f0, read_lo, read_waveform
from Synthese import synthetize
from Noise import synthetize_bruit


def eval(net, file_idx):
    f0_filename = sorted(os.listdir(F0_PATH))[file_idx]
    audio_filename = sorted(os.listdir(AUDIO_PATH))[file_idx]

    f0 = read_f0(f0_filename)
    f0 = torch.tensor(f0).unsqueeze(0).unsqueeze(-1)

    lo = read_lo(audio_filename)
    # drop last frame
    lo = lo[:-1]
    lo = torch.tensor(lo).unsqueeze(0).unsqueeze(-1)

    waveform_truth = read_waveform(audio_filename)

    x = { "f0": f0, "lo": lo }
    with torch.no_grad():
        y = net.forward(x)

    f0 = x["f0"].squeeze(-1)
    a0 = y[:, :, 0]
    aa = y[:, :, 1:NUMBER_HARMONICS + 1]

    if NOISE_ON:
        hs = y[:, :, NUMBER_HARMONICS + 2:NUMBER_HARMONICS + 2 + NUMBER_NOISE_BANDS]
        waveform = synthetize_bruit(a0, f0, aa, hs, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
    else:
        waveform = synthetize(a0, f0, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")

    waveform = waveform.numpy().reshape(-1)
    return waveform, waveform_truth

def test():
    from Net import DDSPNet
    import scipy.io.wavfile

    net = DDSPNet().float()
    net.load_state_dict(torch.load(PATH_TO_CHECKPOINT))
    waveform_synth, waveform_truth = eval(net, file_idx=0)

    scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_syn.wav"), AUDIO_SAMPLE_RATE, waveform_synth)
    scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_ref.wav"), AUDIO_SAMPLE_RATE, waveform_truth)

if __name__ == "__main__":
    test()
