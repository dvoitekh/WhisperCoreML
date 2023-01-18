//
//  Whisper.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import Foundation
import CoreML
import AVFoundation
import Accelerate
import RosaKit



public class Whisper {
    
    // hard-coded audio hyperparameters
    static let kWhisperSampleRate:Int = 16000;
    static let kWhisperNumFFTs:Int = 400;
    static let kWhisperNumMels:Int = 80;
    static let kWhisperHopLength:Int = 160;
    static let kWhisperChunkTimeSeconds:Int = 30;
    // kWhisperChunkTimeSeconds * kWhisperSampleRate  # 480000: number of samples in a chunk
    static let kWhisperNumSamplesInChunk:Int = 480000; // Raw audio chunks we convert to MEL
    // exact_div(kWhisperNumSamplesInChunk, kWhisperHopLength)  # 3000: number of frames in a mel spectrogram input
    static let kWhisperNumSamplesInMel:Int = 3000; // frames of Mel spectrograms

    /// Basic tasks types
    enum WhisperTask
    {
        case Transcribe
        case Translate
    }
    
    // Transcript format - this is the string format of the returned transcript or translation task.
    enum WhisperTranscriptFormat
    {
        case Text // Text only, for Transcription and Translate
        case TextAndTimestamps // Transcription only
        case VTT // Soon Transcription only
        case SRT // Soon Transcription only
    }
    
    // Options to initialize a session with a task, language,
    struct WhisperOptions
    {
        var task:WhisperTask!
        var format:WhisperTranscriptFormat!
        
        // Todo:
        // tempSchedule:[Int]
        // noSpeechTreshold
        //
    }

    /// WhisperSegment internal state tracking for our Whisper session
    // Inspired by https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L153
    struct WhisperSegment
    {
        var id:Int
        var seek:Int
        
        // Segment times in rational time base units
        // Time base is in standard 600 units and isnt in the A
        var startTime:CMTime
        var endTime:CMTime
        
        // Tokens predicted for this session
        var tokens:[Int]
        // Text resultign from decoded tokens
        var decodedText:String
        
        // Todo:
        // temperature
        // avgLogProb
        // compressionRatio
        // noSpeechProb
    }
    
    // MARK:
       
    let decoderModel: decoder_base
    let encoderModel: encoder_base
    let tokenizer = WhisperTokenizer()
    
    let mel:MelSpectrogram = MelSpectrogram(sampleCount: kWhisperNumSamplesInChunk, hopCount: kWhisperHopLength, melCount: kWhisperNumMels, numFFT: kWhisperNumFFTs)

    // a chunk of audio samples, we decode that amount from some input
    // it seems like we pad by 200 in the beginning and end?
    
    // These are variables which cache our current session, tasks and option
    var sessionOptions:WhisperOptions!

    
    var sessionAccruedAudioSamples:[Int16] = []
    var sessionNumAccruedAudioSamples:Int = 0
    var sessionTranscription:[String] = []

    var sessionSegments:[WhisperSegment] = []
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        self.decoderModel = try decoder_base(configuration: config)
        self.encoderModel = try encoder_base(configuration: config)
        
        self.sessionAccruedAudioSamples.reserveCapacity( Whisper.kWhisperNumSamplesInChunk )
        
        self.resetState()
    }
    

    // MARK: Public Methods
    
    /// Call this method whenever you have a new asset, or wish to start a new realtime transcription session
    /// This method resets internal counters, tokens, accrued transcriptions, time stamps, etc
    func startWhisperSession(options:WhisperOptions)
    {
        self.sessionOptions = options
        self.resetState()
    }
    
    
    // this function accrues
    func accrueSamplesFromSampleBuffer(sampleBuffer:CMSampleBuffer)
    {
        
        var audioBufferListSize:Int = 0
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: &audioBufferListSize, bufferListOut: nil, bufferListSize:0, blockBufferAllocator: kCFAllocatorNull, blockBufferMemoryAllocator: kCFAllocatorNull, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: nil)
        
        var audioBufferList = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(audioBufferListSize), mData: nil))

        var blockBuffer:CMBlockBuffer?
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &audioBufferList, bufferListSize: audioBufferListSize, blockBufferAllocator: kCFAllocatorNull, blockBufferMemoryAllocator: kCFAllocatorNull, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: &blockBuffer)
        
        // Determine the number of samples we need from our audio
        
        let numAvailableSamples = Int( CMSampleBufferGetNumSamples(sampleBuffer) )

        // Calculate the number of samples we have to acrrue to get a full chunk
        let remainingSampleCount = Whisper.kWhisperNumSamplesInChunk - self.sessionAccruedAudioSamples.count;
        
        let samplesToAccrue = min(numAvailableSamples, remainingSampleCount);
        
        let remainingCurrentSamplesInBuffer = numAvailableSamples - samplesToAccrue;
        
//        print("numAvailableSamples", numAvailableSamples, "samplesToAccrue", samplesToAccrue, "remainingSampleCount", remainingSampleCount)
        
        let unsafeAudioBufferList = UnsafeMutableAudioBufferListPointer(&audioBufferList)
        
        for (buffer) in unsafeAudioBufferList
        {
            let audioSampleArray:[Int16] = buffer.convertInt16()
                
            let samplesWeNeedToAccrueForAProperChunk = audioSampleArray[0 ... samplesToAccrue - 1]
            
            self.sessionAccruedAudioSamples.insert(contentsOf: samplesWeNeedToAccrueForAProperChunk, at: self.sessionNumAccruedAudioSamples)
                
            self.sessionNumAccruedAudioSamples = self.sessionNumAccruedAudioSamples + samplesWeNeedToAccrueForAProperChunk.count
            
            if (self.sessionAccruedAudioSamples.count == Whisper.kWhisperNumSamplesInChunk)
            {
                do {
                    let encoded = try self.encode(audio: self.sessionAccruedAudioSamples)
                    let transcriptionForChunk:String = try self.decode(audioFeatures: encoded)

                    self.sessionTranscription.append(transcriptionForChunk)
                }
                catch let error
                {
                    
                }
                self.sessionAccruedAudioSamples = []
                self.sessionNumAccruedAudioSamples = 0
            }
            
            
            if (remainingCurrentSamplesInBuffer > 0)
            {
                // Accrue whatever remainder we have..
                print("Remeber to Accrue left over samples")
                
//                let samplesWeNeedToAccrueForAProperChunk = audioSampleArray[remainingCurrentSamplesInBuffer ... samplesToAccrue - 1]

            }
        }

    }
    
    
    func transcribe(assetURL:URL) async -> String
    {
        
        let asset = AVURLAsset(url:assetURL)
        
        do {
            let assetReader = try AVAssetReader(asset: asset)
            
            let audioTracks = try await asset.loadTracks(withMediaType: .audio)
            
            // Output SInt 16
            let audioOutputSettings = [ AVFormatIDKey : kAudioFormatLinearPCM,
                                      AVSampleRateKey : Whisper.kWhisperSampleRate,
                                AVLinearPCMBitDepthKey: 16,
                                 AVNumberOfChannelsKey: 1,
                                AVLinearPCMIsFloatKey : false,
                           AVLinearPCMIsNonInterleaved: false,
                             AVLinearPCMIsBigEndianKey: false
                                        
            ] as [String : Any]
            
            let audioOutput = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: audioOutputSettings)
            audioOutput.alwaysCopiesSampleData = false
            
            if ( assetReader.canAdd(audioOutput) )
            {
                assetReader.add(audioOutput)
            }
            
            assetReader.startReading()
            
            let startTime = NSDate.timeIntervalSinceReferenceDate
            
            while ( assetReader.status == .reading )
            {
                guard let audioSampleBuffer = audioOutput.copyNextSampleBuffer(), CMSampleBufferIsValid(audioSampleBuffer) else {
                    
                    // Some media formats can have weird decode issues.
                    // Unless our asset reader EXPLICITELT tells us its done, keep trying to decode.
                    // We just skip bad samples
                    if ( assetReader.status == .reading)
                    {
                        continue
                    }
                    
                    else if (assetReader.status == .completed)
                    {
                        break;
                    }
                    
                    else
                    {
                        // something went wrong
                        print(assetReader.error as Any)
                        return ""
                    }
                }
                                        
                self.accrueSamplesFromSampleBuffer(sampleBuffer: audioSampleBuffer)
                
            }
                        
            let processingTime = NSDate.timeIntervalSinceReferenceDate - startTime
            
            print("Decode and Predict took", processingTime, "seconds")
            
            let assetDuration = try await asset.load(.duration).seconds
            
            print("Movie is", assetDuration)
            print("Realtime Factor is", assetDuration / processingTime)

            return self.sessionTranscription.joined(separator: " ")
            
        }
        catch let error
        {
            print("Unable to process asset:")
            print(error)
            exit(0)
        }
    }
    
    // MARK: Private Methods
    
    // Internal Helper to just test and visualize the output of our Log Mel processing
    private func normalize(array: [Float]) -> [Float] {
        var normalizedArray = array
           var min = Float.greatestFiniteMagnitude
           var max = -Float.greatestFiniteMagnitude
           var shift: Float = 0.0
           var scale: Float = 0.0
           
           vDSP_minv(array, 1, &min, vDSP_Length(array.count))
           vDSP_maxv(array, 1, &max, vDSP_Length(array.count))
           shift = abs(min)
           vDSP_vsadd(array, 1, &shift, &normalizedArray, 1, vDSP_Length(array.count))
           scale = 1 / (max + shift)
           vDSP_vsmul(normalizedArray, 1, &scale, &normalizedArray, 1, vDSP_Length(array.count))
           return normalizedArray
    }
    
    private func encode(audio: [Int16]) throws -> MLMultiArray {
        // TODO: Fix our vDSP based mel processor
//        let mel:[Float] = mel.processData(audio: audio)

        let mel:[Float] = mel.processDataRosa(audio: audio)
//        let mel = MelSpectrogram.loadReferencePythonRawMelToDebugShit()
        
        let normalizedFloatMel =  self.normalize(array: mel )
        
        normalizedFloatMel.withUnsafeBufferPointer { unsafeMel in
            
            let data = Data(buffer: unsafeMel)
            do {
                try data.write(to: URL(fileURLWithPath: "/Users/vade/Downloads/rawMel-normalized.raw"))
            }
            catch {
            }
        }
        
        let array = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)

        mel.withUnsafeBytes { melPtr in
            array.withUnsafeMutableBytes { arrayPtr, strides in
                memcpy(arrayPtr.baseAddress!, melPtr.baseAddress!, 80 * 3000 * MemoryLayout<Float>.size)
            }
        }
        
        let encoded = try encoderModel.prediction(logmel_data:array).var_719
        return encoded
//        return array
    }
    
    private func decode(audioFeatures: MLMultiArray) throws -> String {
        
        // SOT Initialize sequence
        var tokens:[Int] = []

        // create sot sequence - multilingual model always needs a task and
        // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L325
//         https://github.com/huggingface/transformers/blob/main/tests/models/whisper/test_tokenization_whisper.py
        tokens.append(WhisperTokenizer.sotToken)
        tokens.append(WhisperTokenizer.langToken)
        tokens.append(WhisperTokenizer.transcribeToken)
        
        // No Time Stamps
        if ( self.sessionOptions.format == WhisperTranscriptFormat.Text )
        {
            tokens.append(WhisperTokenizer.notToken)
        }
        
        // Today, we dont support audio frames other than the full 3000 Mel count
        // So seek is count of a mel chunk (3000) * hop length / sample rate
        // See : for reference https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L176
        let segmentFrameCount = Whisper.kWhisperNumSamplesInMel
        
        let seek = self.sessionSegments.count * segmentFrameCount
        
        let timestampOffset = Float64(seek * Whisper.kWhisperHopLength / Whisper.kWhisperSampleRate)
        let segmentDuration =  Float64(segmentFrameCount * Whisper.kWhisperHopLength / Whisper.kWhisperSampleRate)
        
        print("segment start, segment duration", timestampOffset, segmentDuration)
        
        var nextToken = 0

        while ( nextToken != WhisperTokenizer.eotToken )
        {
            autoreleasepool {
 
                let tokensArray = self.tokenizer.tokensToMultiArray(tokens, dims: 2)

                let decoded = try! decoderModel.prediction(token_data: tokensArray, audio_data: audioFeatures).var_1131

                nextToken = self.tokenizer.nextTokenGreedy(decoded: decoded)
                tokens.append(nextToken)
                print(nextToken)
//                let transcription = self.tokenizer.decode(tokens: tokens)
//
//                print(transcription)
            }
        }

        
        let transcription = self.tokenizer.decodeWithTimestamps(tokens: tokens)

        let idForSegment = self.sessionSegments.count
        
        let currentSegment = Whisper.WhisperSegment(id: idForSegment,
                                                    seek: seek,
                                                    startTime: CMTimeMakeWithSeconds(timestampOffset, preferredTimescale: 600),
                                                    endTime: CMTimeMakeWithSeconds(segmentDuration, preferredTimescale: 600),
                                                    tokens: tokens,
                                                    decodedText: transcription)

        self.sessionSegments.append(currentSegment)
        
        print(transcription)
        
        return transcription
    }
    
    private func resetState()
    {
        // Reset our state
        self.sessionSegments = []
        self.sessionAccruedAudioSamples = []
        self.sessionNumAccruedAudioSamples = 0

    }
    
}


// Taken from : https://gist.github.com/tion-low/47e9fc4082717078dff4d6259b6ffbc9

//extension AudioBufferList {
//    public mutating func convert() -> [AudioBuffer] {
//
//        self.mBuffers
//
//        let buf = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer<AudioBufferList>(start: &(self.mBuffers), count: Int(self.mNumberBuffers)) )
//
//        return Array(buf)
//
//
////        let buf: UnsafeBufferPointer<AudioBuffer> = UnsafeBufferPointer<AudioBuffer>(start: &(self.mBuffers), count: Int(self.mNumberBuffers))
////        return
//    }
//}

extension AudioBuffer {
    public func convertFloat() -> [Float] {
        if let mdata = self.mData {
            let ump = mdata.bindMemory(to: Float.self, capacity: Int(mDataByteSize))
            let usp = UnsafeBufferPointer(start: ump, count: Int(mDataByteSize) / MemoryLayout<Float>.size)
            return [Float](usp)
        } else {
            return []
        }
    }
    
    public func convertInt16() -> [Int16] {
        if let mdata = self.mData {
            let ump = mdata.bindMemory(to: Int16.self, capacity: Int(mDataByteSize))
            let usp = UnsafeBufferPointer(start: ump, count: Int(mDataByteSize) / MemoryLayout<Int16>.size)
            return [Int16](usp)
        } else {
            return []
        }
    }

}
