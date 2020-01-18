import scipy.io.wavfile as wav

from Parameters import *
from DataLoader import read_f0, read_lo, read_waveform
from Synthese import synthetize, reverb
from Net import DDSPNet


def evaluation(net, file_idx, device, duration):
    f0_filename = sorted(os.listdir(F0_PATH))[file_idx]
    audio_filename = sorted(os.listdir(AUDIO_PATH))[file_idx]

    f0 = read_f0(f0_filename)
    f0 = torch.tensor(f0[0:duration * FRAME_SAMPLE_RATE])\
        .unsqueeze(0).unsqueeze(-1)

    lo = read_lo(audio_filename)
    lo = lo[:-1]
    lo = lo[0:duration * FRAME_SAMPLE_RATE].unsqueeze(0).unsqueeze(-1)

    waveform_truth = read_waveform(audio_filename)
    waveform_truth = waveform_truth[0:duration * AUDIO_SAMPLE_RATE]

    x = {"f0": f0, "lo": lo}
    with torch.no_grad():
        y_additive, y_bruit = net.forward(x)

    f0 = x["f0"].squeeze(-1)
    a0 = y_additive[:, :, 0]
    aa = y_additive[:, :, 1:NUMBER_HARMONICS + 1]
    hs = y_bruit[:, :, 0:NUMBER_NOISE_BANDS + 1]

    if NOISE_ON:
        additive, noise = synthetize(a0, f0, aa, hs, FRAME_LENGTH,
                                     AUDIO_SAMPLE_RATE, device)
        additive = additive.numpy().reshape(-1)
        noise = noise.numpy().reshape(-1)

        if REVERB:
            additive = reverb(additive)
            noise = reverb(noise)

        waveform = additive + noise

        return additive, noise, waveform, waveform_truth
    else:
        additive, noise = synthetize(a0, f0, aa, hs, FRAME_LENGTH,
                                     AUDIO_SAMPLE_RATE, device)
        waveform = additive
        return waveform, waveform_truth


if __name__ == "__main__":
    ''' Parameters '''
    file_index = 0
    model = "Checkpoint"  # Options : "Full", "Checkpoint"
    working_device = "cpu"  # Use "cpu" when training at the same time
    audio_duration = 20  # Duration of the evaluation in seconds

    ''' Charge net '''
    NET = DDSPNet().float()
    if model == "Full":
        NET.load_state_dict(torch.load(PATH_TO_MODEL, map_location=DEVICE))
    else:
        NET.load_state_dict(torch.load(PATH_TO_CHECKPOINT,
                                       map_location=DEVICE))

    ''' Create and write waveforms '''
    if SEPARED_NOISE:
        ADDITIVE, NOISE, WAVEFORM_SYNTH, WAVEFORM_TRUTH = \
            evaluation(NET, file_idx=file_index, device=working_device,
                       duration=audio_duration)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT +
                               "_additive.wav"), AUDIO_SAMPLE_RATE, ADDITIVE)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT +
                               "_noise.wav"), AUDIO_SAMPLE_RATE, NOISE)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT +
                               "_syn.wav"), AUDIO_SAMPLE_RATE, WAVEFORM_SYNTH)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT +
                               "_ref.wav"), AUDIO_SAMPLE_RATE, WAVEFORM_TRUTH)
    else:
        WAVEFORM_SYNTH, WAVEFORM_TRUTH = evaluation(NET, file_idx=file_index,
                                                    device=working_device,
                                                    duration=audio_duration)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_syn.wav"),
                  AUDIO_SAMPLE_RATE, WAVEFORM_SYNTH)
        wav.write(os.path.join("Outputs", "Eval_" + INSTRUMENT + "_ref.wav"),
                  AUDIO_SAMPLE_RATE, WAVEFORM_TRUTH)
