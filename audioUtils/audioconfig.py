# Audio config values to be used for the rest of the code

import pyaudio

THRESHOLD = 150  # Originally 500, but new is_silent() works better with this value
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
