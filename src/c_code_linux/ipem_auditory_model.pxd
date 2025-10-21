# ipem_auditory_model.pxd

cdef extern from "IPEMAuditoryModel.h":
    int IPEMAuditoryModel_Setup(
        long inNumOfChannels,
        double inFirstFreq,
        double inFreqDist,
        const char* inInputFileName,
        const char* inInputFilePath,
        const char* inOutputFileName,
        const char* inOutputFilePath,
        double inSampleFrequency,
        long inSoundFileFormat
    )

    long IPEMAuditoryModel_Process()