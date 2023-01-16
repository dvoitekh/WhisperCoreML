
# Whisper CoreML

A port of OpenAI's Whisper Speech Transcription model to CoreML

The goal of this project is to natively port, and optimize Whisper for use on Apple Silicon including optimization for the Apple Neural Engine, and match the incredible WhisperCPP project on features.

You can:

Create a Whipser instance `whisper = try Whisper()`

And run transcription on a Quicktime compatible asset via: `await whisper.predict(assetURL: url)`

And Whipser CoreML will load an asset using AVFoundation and convert the audio to the appropriate format for transcription.

Alternatively, for realtime usage, you can call `accrueSamplesFromSampleBuffer(sampleBuffer:CMSampleBuffer)` and vend samples from an AVCaptureSession or AVAudioSession. Note, we accrue a 30 second sample for now as that is the expected number of samples required. 

## Status
* [X] Working Multi Lingual Transcription
* [ ] [Optimize the CoreML models for ANE](https://machinelearning.apple.com/research/neural-engine-transformers) using [Apples ANE Transformers sample code found at this repository](https://github.com/apple/ml-ane-transformers)
* [ ] Port Log Mel Spectrogram to native vDSP and ditch RosaKit package dependency.
* [ ] Decode Special Tokens for time stamps.
* [ ] Decide on API design

## Performance

* Base model gets roughly 4x realtime using a single core on an M1 Mac Book Pro.


## Getting Models:

[For ease of use, you can use this Google Colab to convert models](https://colab.research.google.com/drive/1IiBx6-hipt3ER3VjkjuUEAObwipHy1mL
). Note that if you convert Medium or larger models you may run into memory issues on Google Colab. 

This repository assumes youre converting multilingual models. If you need 'en' models you'll need to adjust the special token values by negative 1.
