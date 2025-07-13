
# Anatomy of a WAV File

WAV files follow the RIFF (Resource Interchange File Format) specification. A file consists of one or more **chunks**, each with an identifier, size, and associated data. The most important chunks are:

## RIFF Chunk (Header)

| Field            | Size (Bytes) | Description                     |
|------------------|--------------|---------------------------------|
| Chunk ID         | 4            | Always `"RIFF"`                 |
| Chunk Size       | 4            | File size minus 8 bytes         |
| Format           | 4            | Always `"WAVE"`                 |

## `fmt` Chunk (Standard)

| Field              | Size (Bytes) | Description                         |
|--------------------|--------------|-------------------------------------|
| Chunk ID           | 4            | `"fmt "` (note the trailing space)  |
| Chunk Size         | 4            | Size of the rest of the chunk       |
| Audio Format       | 2            | 1 = PCM, 3 = IEEE float             |
| Num Channels       | 2            | Mono = 1, Stereo = 2, etc.          |
| Sample Rate        | 4            | Hz                                  |
| Byte Rate          | 4            | SampleRate × NumChannels × Bits/8   |
| Block Align        | 2            | NumChannels × BitsPerSample / 8     |
| Bits Per Sample    | 2            | 16, 24, 32, 64 (Wavers does not support 8-bit) |

## `fmt` Chunk (Extended: WAVE_FORMAT_EXTENSIBLE)

| Field                      | Size (Bytes) | Description                                           |
|----------------------------|--------------|-------------------------------------------------------|
| Extension Size             | 2            | Always 22 for WAVE_FORMAT_EXTENSIBLE                 |
| Valid Bits Per Sample     | 2            | Bits of precision in the sample data                 |
| Channel Mask               | 4            | Speaker position mask (e.g., front left, center)     |
| Subformat GUID             | 16           | Defines actual format (PCM, IEEE float, etc.)        |

## `data` Chunk

| Field         | Size (Bytes) | Description                    |
|---------------|--------------|--------------------------------|
| Chunk ID      | 4            | `"data"`                       |
| Chunk Size    | 4            | Size of audio data in bytes    |
| Audio Data    | variable     | Raw sample data                |

Other chunks (e.g., `LIST`, `fact`, `PEAK`) may also appear in extended files. Wavers will safely ignore unsupported chunks and can be extended to process custom metadata in the future. Currently, WaveRS does not write these extra chunks back to the file when writing. 
