import os
import torch
import torchaudio
from speechbrain.inference import Tacotron2
from speechbrain.inference import HIFIGAN
import re

class TTSEngine:
    def __init__(self):
        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_models/tts-tacotron2")
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan")

    def generate_speech(self, text, output_file):
        # Split the text into sentences
        sentences = re.split('(?<=[.!?]) +', text)
        all_waveforms = []

        for sentence in sentences:
            # Generate mel spectrogram
            mel_output, mel_length, alignment = self.tacotron2.encode_text(sentence)
            
            # Generate waveform
            waveforms = self.hifi_gan.decode_batch(mel_output)
            all_waveforms.append(waveforms[0])

        # Concatenate all waveforms
        final_waveform = torch.cat(all_waveforms, dim=1)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the audio
        torchaudio.save(output_file, final_waveform.cpu(), self.tacotron2.hparams.sample_rate)
        return output_file

tts_engine = TTSEngine()