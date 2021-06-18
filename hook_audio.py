import base64
import io
import logging

import librosa
from ml_serving.utils import helpers
import numpy as np
import soundfile as sf

from synthesizer.inference import Synthesizer
from encoder import inference as enc_inference
from vocoder import inference as voc_inference


LOG = logging.getLogger(__name__)
PARAMS = {
    'vocoder': None,
    'encoder': None,
    'synthesizer': None,
}


class AudioGen:
    def __init__(self, encoder, vocoder, synthesizer):
        self.synthesizer = Synthesizer(synthesizer)
        enc_inference.load_model(encoder)
        voc_inference.load_model(vocoder)

        self.speaker_embedding_size = 256

        self.test()

    def test(self):
        print("\tTesting the encoder...")
        enc_inference.embed_utterance(np.zeros(enc_inference.sampling_rate))

        embed = np.random.rand(self.speaker_embedding_size)
        embed /= np.linalg.norm(embed)
        embeds = [embed, np.zeros(self.speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        print("\tTesting the synthesizer... (loading the model will output a lot of text)")
        mels = self.synthesizer.synthesize_spectrograms(texts, embeds)

        mel = np.concatenate(mels, axis=1)
        no_action = lambda *args: None
        print("\tTesting the vocoder...")
        voc_inference.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
        print("All test passed! You can now synthesize speech.\n\n")

    def generate_speech(self, speech_example_bytes, text):
        bytes_io = io.BytesIO(speech_example_bytes)

        original_wav, sampling_rate = librosa.load(bytes_io)
        preprocessed_wav = enc_inference.preprocess_wav(original_wav, sampling_rate)

        embed = enc_inference.embed_utterance(preprocessed_wav)

        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        def progress(i, seq_len, b_size, gen_rate):
            pass
        generated_wav = voc_inference.infer_waveform(spec, progress_callback=progress)

        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = enc_inference.preprocess_wav(generated_wav)

        audio_bytes = io.BytesIO()
        # sf.write('test.wav', generated_wav.astype(np.float32), self.synthesizer.sample_rate)
        sf.write(
            audio_bytes,
            generated_wav.astype(np.float32),
            self.synthesizer.sample_rate,
            format='wav'
        )
        return audio_bytes.getvalue()


def init_hook(ctx, **params):
    PARAMS.update(params)
    audio_gen = AudioGen(
        encoder=PARAMS['encoder'],
        vocoder=PARAMS['vocoder'],
        synthesizer=PARAMS['synthesizer'],
    )
    return audio_gen


def process(inputs, ctx, **kwargs):
    audio_gen: AudioGen = ctx.global_ctx

    audio_bytes = helpers.get_param(inputs, 'audio', raw_bytes=True)
    if isinstance(audio_bytes, str):
        audio_bytes = base64.decodebytes(audio_bytes.encode())
    text = helpers.get_param(inputs, 'text')

    generated_speech = audio_gen.generate_speech(audio_bytes, text)
    result = {'audio': generated_speech}
    return result
