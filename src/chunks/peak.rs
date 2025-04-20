/// The PEAK chunk

/// This document describes a soundfile information chunk that may be added to AIFF, AIFF-C and WAVE soundfiles to indicate to peak amplitude of the soundfile. This information can be used to facilitate soundfile display, quick normalization, and the playback of floating point files.

/// The PEAK chunk structure:

/// typedef struct
/// {
///     char          chunkID[4];     /* 'PEAK' */
///     unsigned long chunkDataSize;  /* the size of the chunk */
///     unsigned long version;        /* version of the PEAK chunk */
///     unsigned long timeStamp;      /* secs since 1/1/1970  */
///     PositionPeak  peak[CommonChunk.numchannels]; /* the peak info */
/// }   PeakChunk;

/// <chunkID> is the character array 'PEAK', for chunk identification.

/// <chunkDataSize> is the size of the data section of the chunk. It does
///  not include the 8 bytes used by chunkID and chunkDataSize.

/// <version> indicates the version of the PEAK chunk. This allows anything
///  after PeakChunk.version to be redefined for future needs. This document
///  describes version 1.

/// <timeStamp> is the number of seconds since 1/1/1970. This is used to
///  see if the date of the peak data matches the modification date of
///  the file. If not, the file should be rescanned for new peak data.

/// <peak> is a PositionPeak structure. There is a PositionPeak structure
///  for every channel. These are in the same channel order as the samples
///  are interleaved.

/// The PositionPeak structure for version 1:

/// typedef struct
/// {
///     float         value;    /* signed value of peak */
///     unsigned long position; /* the sample frame for the peak */
/// }   PositionPeak;

/// <value> is the signed peak value. A value of 1.0 represents 0dBFS for
///  non floating point encodings.

/// <position> is the sample frame number at which the peak occurs. Note
///  that the unit for position are sample frames, not sample points nor
///  bytes.