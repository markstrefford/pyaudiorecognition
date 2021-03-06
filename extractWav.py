# (1)
import wave
import sys
import struct
import numpy

# (2)
# Get wav filename
if len(sys.argv) == 2:
    wav_file_name = sys.argv[1]
else:
    wav_file_name = "output.wav"

# (3)
# Check file exists by opening it to read
wav_file = wave.open(wav_file_name, "r")

# (4)
# Display some data about the recorded sound
frame_rate = wav_file.getframerate()
num_channels = wav_file.getnchannels()
num_frames = wav_file.getnframes()

frames = wav_file.readframes(num_frames)
data = numpy.array(struct.unpack_from("%dh" % wav_file.getnframes(), frames))

# (5)
print ".wav Filename: " + wav_file_name
print "Frame rate: " + str(frame_rate)
print "Channels: " + str(num_channels)
print "Number of frames:" + str(num_frames)
print "Data size:", data.shape
# (6)

wav_file.close()
