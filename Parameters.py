#### PARAMETERS FILE ####

import os
import torch

## FOLDER PARAMETERS ##
INSTRUMENT = "Sax"
AUDIO_PATH = os.path.join("Inputs", INSTRUMENT, "Audio")
F0_PATH = os.path.join("Inputs", INSTRUMENT, "F0")
FRAGMENT_CACHE_PATH = os.path.join("Cache", INSTRUMENT)
FRAGMENT_CACHE_PATTERN = "{:d}.pth"


## MODELS ##
PATH_SAVED_MODELS = os.path.join("Models")
MODEL_NAME = "Model_" + INSTRUMENT
MODEL_CHECKPOINT = MODEL_NAME + "_checkpt"
PATH_TO_CHECKPOINT = os.path.join(PATH_SAVED_MODELS, MODEL_CHECKPOINT + ".pth")
PATH_TO_MODEL = os.path.join(PATH_SAVED_MODELS, MODEL_NAME + ".pth")


## TRAIN PARAMETERS ##
SHUFFLE_DATALOADER = True
BATCH_SIZE = 3
FFT_SIZES = [2048, 1024, 512, 256, 128, 64]
NUMBER_EPOCHS = 500
LEARNING_RATE = 0.001
SCHEDULER_RATE = 0.99

GPU_ON = True
CUDA_ON = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_ON and GPU_ON else "cpu")


## DATA PARAMETERS ##
AUDIO_SAMPLE_RATE = 16000
FRAME_SAMPLE_RATE = 100  # Hertz
FRAME_LENGTH = AUDIO_SAMPLE_RATE // FRAME_SAMPLE_RATE
AUDIOFILE_DURATION = 60  # Seconds
FRAGMENT_DURATION = 2  # Seconds
FRAGMENTS_PER_FILE = int(AUDIOFILE_DURATION / FRAGMENT_DURATION)
SAMPLES_PER_FRAGMENT = FRAGMENT_DURATION * FRAME_SAMPLE_RATE


## NET PARAMETERS ##
LINEAR_OUT_DIM = 512
LINEAR_ADDITIVE_DIM = 256
LINEAR_NOISE_DIM = 256
assert LINEAR_ADDITIVE_DIM + LINEAR_NOISE_DIM <= LINEAR_OUT_DIM
HIDDEN_DIM = 512
NUMBER_HARMONICS = 64
NUMBER_NOISE_BANDS = 65
NOISE_ON = True
SEPARED_NOISE = True


## SYNTHESIS PARAMETERS ##
SYNTHESIS_DURATION = 2  # Seconds
SYNTHESIS_SAMPLING_RATE = 16000  # Hertz
HAMMING_WINDOW_LENGTH = 128
HANNING_SMOOTHING = True
