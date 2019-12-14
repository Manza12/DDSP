from Parameters import *
from DataLoader import read_f0, read_lo, read_waveform
from Synthese import synthetize


def eval(net, file_idx):
    f0_filename = sorted(os.listdir(RAW_PATH))[file_idx]
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
    aa = y[:, :, 1:]

    waveform = synthetize(a0, f0, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
    waveform = waveform.numpy().reshape(-1)
    return waveform, waveform_truth

def test():
    from Net import DDSPNet
    import scipy.io.wavfile

    net = DDSPNet().float()
    net.load_state_dict(torch.load(os.path.join("Models", "Model_with_Olivier_checkpoint.pth")))
    waveform_synth, waveform_truth = eval(net, file_idx=0)

    scipy.io.wavfile.write(os.path.join("Outputs", "eval_syn.wav"), AUDIO_SAMPLE_RATE, waveform_synth)
    scipy.io.wavfile.write(os.path.join("Outputs", "eval_ref.wav"), AUDIO_SAMPLE_RATE, waveform_truth)

if __name__ == "__main__":
    test()
