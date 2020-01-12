from Parameters import *
from DataLoader import read_f0, read_lo, read_waveform
from Synthese import synthetize_smooth, synthetize
from Noise import synthetize_bruit, synthetize_additive_plus_bruit
from Net import DDSPNet
import scipy.io.wavfile


def evaluation(net, file_idx):
    f0_filename = sorted(os.listdir(F0_PATH))[file_idx]
    audio_filename = sorted(os.listdir(AUDIO_PATH))[file_idx]

    f0 = read_f0(f0_filename)
    f0 = torch.tensor(f0).unsqueeze(0).unsqueeze(-1)

    lo = read_lo(audio_filename)
    lo = lo[:-1]
    lo = torch.tensor(lo).unsqueeze(0).unsqueeze(-1)

    waveform_truth = read_waveform(audio_filename)

    x = { "f0": f0, "lo": lo }
    with torch.no_grad():
        y_additive, y_bruit = net.forward(x)

    f0 = x["f0"].squeeze(-1)
    a0 = y_additive[:, :, 0]
    aa = y_additive[:, :, 1:NUMBER_HARMONICS + 1]

    if NOISE_ON:
        hs = y_bruit[:, :, 0:NUMBER_NOISE_BANDS + 1]
        h0s = y_bruit[:, :, 0]
        if SEPARED_NOISE:
            additive, noise = synthetize_additive_plus_bruit(a0, f0, aa, hs, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
            additive = additive.numpy().reshape(-1)
            noise = noise.numpy().reshape(-1)
            waveform = additive + noise
            return additive, noise, waveform, waveform_truth
        else:
            waveform = synthetize_bruit(a0, f0, aa, hs, h0s, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
            waveform = waveform.numpy().reshape(-1)
            return waveform, waveform_truth
    else:
        if HANNING_SMOOTHING:
            waveform = synthetize_smooth(a0, f0, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
            return waveform, waveform_truth
        else:
            waveform = synthetize(a0, f0, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE, "cpu")
            return waveform, waveform_truth


if __name__ == "__main__":
    #### File data ####
    file_index = 0

    #### Charge net ####
    NET = DDSPNet().float()
    NET.load_state_dict(torch.load(PATH_TO_CHECKPOINT))

    #### Create and write waveforms ####
    if SEPARED_NOISE:
        ADDITIVE, NOISE, WAVEFORM_SYNTH, WAVEFORM_TRUTH = evaluation(NET, file_idx=file_index)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_additive.wav"), AUDIO_SAMPLE_RATE,
                               ADDITIVE)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_noise.wav"), AUDIO_SAMPLE_RATE,
                               NOISE)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_syn.wav"), AUDIO_SAMPLE_RATE,
                               WAVEFORM_SYNTH)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_ref.wav"), AUDIO_SAMPLE_RATE,
                               WAVEFORM_TRUTH)
    else:
        WAVEFORM_SYNTH, WAVEFORM_TRUTH = evaluation(NET, file_idx=file_index)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_syn.wav"), AUDIO_SAMPLE_RATE,
                               WAVEFORM_SYNTH)
        scipy.io.wavfile.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_ref.wav"), AUDIO_SAMPLE_RATE,
                               WAVEFORM_TRUTH)
