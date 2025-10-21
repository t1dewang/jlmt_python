# ipem_auditory_model.pyx
# distutils: language = c
# cython: language_level=3

cdef extern from "IPEMAuditoryModel.h":
    void IPEMAuditoryModel_Setup(
        long inNumOfChannels,
        double inFirstFreq,
        double inFreqDist,
        const char* inInputFileName,
        const char* inInputFilePath,
        const char* inOutputFileName,
        const char* inOutputFilePath,
        double inSampleFrequency,
        long inSoundFileFormat)

    long IPEMAuditoryModel_Process()

def process_file(
    input_filename,
    output_filename,
    num_channels=-1,
    first_freq=-1.0,
    freq_dist=-1.0,
    input_filepath=".",
    output_filepath=".",
    sample_frequency=-1.0,
    sound_format="wav"
):

    cdef bytes b_input_filename = input_filename.encode('utf-8')
    cdef bytes b_output_filename = output_filename.encode('utf-8')
    cdef bytes b_input_filepath = input_filepath.encode('utf-8')
    cdef bytes b_output_filepath = output_filepath.encode('utf-8')

    cdef const char* c_input_filename = b_input_filename
    cdef const char* c_output_filename = b_output_filename
    cdef const char* c_input_filepath = b_input_filepath
    cdef const char* c_output_filepath = b_output_filepath

    cdef long c_sound_format = 0
    if sound_format.lower() == "snd":
        c_sound_format = 1  # sffSnd
    else:
        c_sound_format = 0  # sffWav

    IPEMAuditoryModel_Setup(
        num_channels,
        first_freq,
        freq_dist,
        c_input_filename,
        c_input_filepath,
        c_output_filename,
        c_output_filepath,
        sample_frequency,
        c_sound_format
    )

    result = IPEMAuditoryModel_Process()
    return int(result)
