import base64
import io
import logging
import re
import subprocess
import time
import uuid
import tempfile

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
SR_REGEX = re.compile('([0-9]+) Hz')


class CachedVector:
    def __init__(self, vector, time):
        self.vector = vector
        self.time = time


class AudioGen:
    def __init__(self, encoder, vocoder, synthesizer):
        self.synthesizer = Synthesizer(synthesizer)
        enc_inference.load_model(encoder)
        voc_inference.load_model(vocoder)

        self.speaker_embedding_size = 256
        self.keep_cache_sec = 3600
        self.cache = {}

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

    def _save_tmp(self, bytes_):
        fname = tempfile.mktemp(suffix='.wav')
        with open(fname, 'wb') as f:
            f.write(bytes_)

        return fname

    def _get_sample_rate(self, fname):
        pipe = subprocess.Popen(
            ['ffprobe', '-hide_banner', fname],
            stderr=subprocess.PIPE,
        )
        output = pipe.stderr.read().decode()
        pipe.communicate()
        groups = re.findall(SR_REGEX, output)
        if len(groups) < 1:
            raise RuntimeError('Can not estimate sample rate for file. Please ensure that file is correct.')

        return int(groups[0])

    def get_embedding(self, speech_example_bytes):
        fname = self._save_tmp(speech_example_bytes)
        sample_rate = self._get_sample_rate(fname)

        original_wav, sampling_rate = librosa.load(fname, sr=sample_rate)
        preprocessed_wav = enc_inference.preprocess_wav(original_wav, sampling_rate, trim_silence=False)

        embed = enc_inference.embed_utterance(preprocessed_wav)
        return embed

    def synthesize(self, embed, text):
        embeds = [embed]
        texts = [text]
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

    def generate_speech(self, speech_example_bytes, text):
        embed = self.get_embedding(speech_example_bytes)

        return self.synthesize(embed, text)

    def cache_vector(self, vector):
        vector_id = str(uuid.uuid4())
        LOG.info(f'[Cache] New vector ID={vector_id}')
        self.cache[vector_id] = CachedVector(vector, time.time())
        self._invalidate_cache()
        return vector_id

    def get_cached_vector(self, key):
        result = self.cache[key]
        self._invalidate_cache()
        return result.vector

    def _invalidate_cache(self):
        keys = list(self.cache.keys())
        now = time.time()
        for key in keys:
            if now - self.cache[key].time >= self.keep_cache_sec:
                LOG.info(f'[Cache] Expired vector ID={key}')
                del self.cache[key]


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

    speech_id = helpers.get_param(inputs, 'speech_id')
    generate_speech = helpers.boolean_string(
        helpers.get_param(inputs, 'generate_speech', default=True)
    )
    audio_bytes = helpers.get_param(inputs, 'audio', raw_bytes=True)
    if isinstance(audio_bytes, str):
        audio_bytes = base64.decodebytes(audio_bytes.encode())
    text = helpers.get_param(inputs, 'text')

    if audio_bytes is not None and len(audio_bytes) > 0:
        vector = audio_gen.get_embedding(audio_bytes)
        speech_id = audio_gen.cache_vector(vector)
    else:
        vector = audio_gen.get_cached_vector(speech_id)

    result = {'speech_id': speech_id}

    # Generate speech from speech_id
    if generate_speech:
        generated_speech = audio_gen.synthesize(vector, text)
        result['audio'] = generated_speech

    return result
