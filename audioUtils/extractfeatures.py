#
# Extract voice features
#
# Based on https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import wave
import sys
import numpy
import scipy.fftpack
import struct
import math
from sklearn import preprocessing
from audioUtils import audioconfig as config

# Determine number of frames in the data
def getnframes(data):
    return (len(data)-2)/4      # Remove string quotes, then assume 4 bytes per frame

# Get frames from provided data
# This returns the speech frames as a numpy array
def get_frames(speech_data, frame_length, frame_shift, frame_count):
    # Ensure we have a complete final frame by zero padding
    zeropad_size = speech_data.shape[0] % frame_length
    speech_data_zero_padded = numpy.append(speech_data, numpy.zeros( (frame_length-zeropad_size), ) )
    speech_frames =[]
    frame_start_position = 0
    while True:
        windowed_frame = speech_data_zero_padded[frame_start_position:frame_start_position+frame_length]* numpy.hamming(frame_length)
        speech_frames.append(windowed_frame)
        frame_start_position = frame_start_position + frame_shift
        if frame_start_position+frame_length >= speech_data_zero_padded.shape[0]:
            break
    return numpy.array(speech_frames)

def gen_mel_filts(num_filts, framelength, samp_freq):
    mel_filts = numpy.zeros((framelength, num_filts))
    step_size = int(framelength/float((num_filts + 1))) #Sketch it out to understand
    filt_width = math.floor(step_size*2)
    filt = numpy.bartlett(filt_width)
    step = 0
    for i in xrange(num_filts):
        mel_filts[step:step+filt_width, i] = filt
        step = step + step_size
    # Let's find the linear filters that correspond to the mel filters
    # The freq axis goes from 0 to samp_freq/2, so...
    samp_freq = samp_freq/2
    filts = numpy.zeros((framelength, num_filts))
    for i in xrange(num_filts):
        for j in xrange(framelength):
            freq = (j/float(framelength)) * samp_freq
            # See which freq pt corresponds on the mel axis
            mel_freq = 1127 * numpy.log( 1 + freq/700  )
            mel_samp_freq = 1127 * numpy.log( 1 + samp_freq/700  )
            # where does that index in the discrete frequency axis
            mel_freq_index = int((mel_freq/mel_samp_freq) * framelength)
            if mel_freq_index >= framelength-1:
                mel_freq_index = framelength-1
            filts[j,i] = mel_filts[mel_freq_index,i]
    # Let's normalize each filter based on its width
    for i in xrange(num_filts):
        nonzero_els = numpy.nonzero(filts[:,i])
        width = len(nonzero_els[0])
        filts[:,i] = filts[:,i]*(10.0/width)
    return filts

# # Get wav filename
# if len(sys.argv) == 2:
#     wav_file_name = sys.argv[1]
# else:
#     wav_file_name = "output.wav"
#
# # Check file exists by opening it to read
# wav_file = wave.open(wav_file_name, "r")
#
# # Display some data about the recorded sound
# frame_rate = wav_file.getframerate()
# num_channels = wav_file.getnchannels()
# num_frames = wav_file.getnframes()
#
# print ".wav Filename: " + wav_file_name
# print "Frame rate: " + str(frame_rate)
# print "Channels: " + str(num_channels)
# print "Number of frames:" + str(num_frames)
#
# wavFrames = wav_file.readframes(wav_file.getnframes())

def extractfeatures(data):

    frame_rate = config.RATE
    num_frames = getnframes(data)

    # Set up parameters ready for extracting features
    # Use ones from https://www.youtube.com/watch?v=N34sNSjB04M initially
    # Note that these are frames that we'll use, so are not the same frames as in the audio (these here are less granular!)

    frame_size = 0.025 #in seconds
    frame_shift = 0.0125 #in seconds
    frame_length = int(frame_rate * frame_size)
    frame_shift_length = int(frame_rate * frame_shift)
    frame_count = num_frames / frame_length

    # Extract individual frames from the audio (frame_size seconds in length)
    # and perform discrete fourier transformation of the audio
    speech_frames = get_frames(data, frame_length, frame_shift_length, frame_count)
    audio_spectrum = abs(numpy.fft.rfft(speech_frames,n=1024))  # 1024 = frames per buffer from recording??
    #numpy.savetxt('audio_spectrum.data', audio_spectrum)

    # Mel warping?
    # The mel frequency spectrum aligns more closely with the human ear's perception of sound
    # See https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
    #filts = gen_mel_filts(40, 513, frame_rate)
    spectrum_frame_length = audio_spectrum.shape[1]
    filts = gen_mel_filts(40, spectrum_frame_length, frame_rate)
    mel_spec = numpy.dot(audio_spectrum, filts)  # Dot product of the mel_filts and the audio_spectrum. This does what?
    #numpy.savetxt('mel_spec.data', mel_spec)

    # Now create mel log spectrum
    mel_log_spec = mel_spec # Trust the original author??!!
    nonzero = mel_log_spec > 0
    mel_log_spec[nonzero] = numpy.log(mel_log_spec[nonzero])

    # Mel Cepstrum
    mel_cep = scipy.fftpack.dct(mel_log_spec)
    #numpy.savetxt('mel_cep_all.data', mel_cep)

    mel_cep = mel_cep[:,0:13]
    #numpy.savetxt('mel_cep_0-13.data', mel_cep)

    # Calculate mel_cep deltas
    mel_cep_shift = numpy.delete(mel_cep,[0,1],axis=0)
    blanks = numpy.zeros((2,mel_cep_shift.shape[1]))
    mel_cep_shift = numpy.append(mel_cep_shift, blanks, axis=0)

    mel_cep_deltas = mel_cep_shift - mel_cep
    all_feats = numpy.append(mel_cep,mel_cep_deltas, axis=1)

    #Mel Cep Delta-deltas
    mel_cep_shift = numpy.delete(mel_cep_deltas,[0,1],axis=0)
    mel_cep_shift = numpy.append(mel_cep_shift, blanks, axis=0)
    mel_cep_delta_deltas = mel_cep_shift - mel_cep_deltas
    all_feats = numpy.append(all_feats, mel_cep_delta_deltas, axis=1)

    # Cepstral Mean and Variance Normalization
    all_feats_norm = preprocessing.scale(all_feats)

    #numpy.savetxt('mel_cep.data', all_feats_norm)
    return all_feats_norm

