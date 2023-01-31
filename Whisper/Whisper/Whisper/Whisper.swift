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

protocol WhisperLogitFilter
{
    //  Parameters
    //    ----------
    //    logits : Tensor, shape = (n_batch, vocab_size)
    //        per-token logits of the probability distribution at the current step
    //    tokens : Tensor, shape = (n_batch, current_sequence_length)
    //        all tokens in the context so far, including the prefix and sot_sequence tokens
    func apply(logits: inout MLShapedArray<Float>, tokens: inout MLShapedArray<Int32>)
}

public class Whisper {
    
    // MARK: Public Constants Enums and Structs
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

    enum WhisperError:Error
    {
        case notImplementedYet // Just havent gotten there hang tight.
        case unrecoverableError
    }
    
    /// Basic tasks types
    enum WhisperTask
    {
        case Transcribe
        case Translate
    }
    
    /// Transcript format - this is the string format of the returned transcript or translation task.
    enum WhisperTranscriptFormat
    {
        /// Output text only - Transcription or Translation
        case Text
        /// Output text with timestamps - suitable for Transcription only
        case TextAndTimestamps
        /// Soon - Transcript as VTT formatted text
        case VTT
        /// Soon - Transcript as SRT formatted text
        case SRT
    }
    
    /// Options to initialize a session with a task, language,
    /// See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L19
    struct WhisperOptions
    {
        var task:WhisperTask!
        var format:WhisperTranscriptFormat!
        
        var verbose = false
        
        // Below are WIP

        /// Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        /// upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
        var temperatureSchedule:[Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        /// If the gzip compression ratio is above this value, treat as failed
        var compressionRatioTresh:Float = 2.4
        /// If the average log probability over sampled tokens is below this value, treat as failed
        var logProbThresh:Float = -1.0
        /// If the no_speech probability is higher than this value AND the average log probability
        /// over sampled tokens is below `logprob_threshold`, consider the segment as silent
        var noSpeechThresh:Float = 0.6
        
        /// if True, the previous output of the model is provided as a prompt for the next window;
        /// disabling may make the text inconsistent across windows, but the model becomes less prone to
        /// getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
        var conditionOnPrevText = true
    }

    // MARK: Private Constants Enums and Structs
    
    // All of these are major WIP
   
    // See https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L383
    private struct SupressBlank: WhisperLogitFilter
    {
        let tokenizer:WhisperTokenizer!
        let encodedBlank:[Int]!
        let sampleBegin:Int!
        
        init(tokenizer: WhisperTokenizer!, sampleBegin: Int) {
            self.tokenizer = tokenizer
            self.encodedBlank = tokenizer.encode(text: " ")
            self.sampleBegin = sampleBegin
        }
        
        func apply(logits: inout MLShapedArray<Float>, tokens: inout MLShapedArray<Int32>)
        {
            print("Not Yet Implemented")
            
            // https://www.geeksforgeeks.org/how-to-slice-a-3d-tensor-in-pytorch/
            // tensor[tensor_position_start:tensor_position_end, tensor_dimension_start:tensor_dimension_end , tensor_value_start:tensor_value_end]
            // logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            
            // Need to - for all batch dimensions, access tokens at the encoded " " int value, and the to value and set to -Inf
            // Float.infinity.negate()
        }
    }
    
    private struct SupressTokens: WhisperLogitFilter
    {
        let suppress:[Int]
        
        init(suppress: [Int]) {
            self.suppress = suppress
        }
        func apply(logits: inout MLShapedArray<Float>, tokens: inout MLShapedArray<Int32>)
        {
            print("Not Yet Implemented")
        }
    }
    
    private struct ApplyTimestampRules: WhisperLogitFilter
    {
        let tokenizer:WhisperTokenizer!
        let sampleBegin:Int!
        let maxInitialTimestampIdx:Int?
        
        init(tokenizer: WhisperTokenizer!, sampleBegin: Int!, maxInitialTimestampIdx: Int?) {
            self.tokenizer = tokenizer
            self.sampleBegin = sampleBegin
            self.maxInitialTimestampIdx = maxInitialTimestampIdx
        }

        func apply(logits: inout MLShapedArray<Float>, tokens: inout MLShapedArray<Int32>)
        {
            print("Not Yet Implemented")
        }

    }
    
    
    // WhisperSegment internal state tracking for our Whisper session
    // See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L153
    private struct WhisperSegment
    {
        var id:Int!
        var seek:Int!
        
        // Segment times in rational time base units
        // Time base is in standard 600 units
        var startTime:CMTime!
        var endTime:CMTime!
        
        // Tokens predicted for this segment
        var textTokens:[Int]!
        // Text resulting from decoded tokens
        var decodedText:String!

        // Todo:
        var temperature:Float!
        var avgLogProb:Float!
        var compressionRatio:Float!
        var noSpeechProb:Float!
    }
    
    private enum WhisperDecodingStrategy
    {
        case Greedy
        case BeamSearch // Not implemented yet
    }
    
    // Vended by the decode method and used internally
    // See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py
    private struct WhisperDecodingOptions
    {
        var task:WhisperTask
        var languageCode:String?
    
        var decodingStetegy:WhisperDecodingStrategy = .Greedy
        
        // FYI Semantics from Python are these values can be
        // None
        // Zero
        // Some value
        // each has specific meaning, specifically None.
        // We treat optional / nil as none here.
        
        // Sampling Related Options
        
        var temperature:Float = 0.0
        // Maximum number of tokens to sample
        var maxSampleLen:Int?
        // Number of independent samples to collect, when t > 0
        var bestOf:Int?
        // number of beams in beam search, when t == 0
        var beamSize:Int?
        // patience in beam search (https://arxiv.org/abs/2204.05424)
        var patience:Float?
        
        // Options for ranking generations (either beams or best-of-N samples)
        
        // "alpha" in Google NMT, None defaults to length norm
        var lengthPenalty:Float?
        
        // Prompt, prefix, and token suppression
        
        // Text or tokens for the previous context
        var prompt:String?
        var promptTokens:[Int]?
        // text or tokens to prefix the current context
        var prefix:String?
        var prefixTokens:[Int]?
        // this will suppress blank outputs
        var suppressBlank:Bool = true
        // list of tokens ids (or comma-separated token ids) to suppress
        // nil will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
        // and empty array will do no suppression
        var suppresTokens:[Int]?
        
        // timestamp sampling options
        var withoutTimestamps:Bool = false
        var maxInitialTimestampL:Float = 1.0
    }
    
    // https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L104
    private struct WhisperDecodingResult
    {
        var tokens:[Int]
        var text:String = ""

        var languageCode:String?
        var langProbs:[String:Float]?
        
        var avgLogProbs:Float = Float.nan
        var noSpeechProbs:Float = Float.nan
        var temperature:Float = Float.nan
        var compressionRatio:Float = Float.nan
    }
    
    // MARK: Whisper Properties
       
    let decoderModel: decoder_base
    let encoderModel: encoder_base
    let tokenizer = WhisperTokenizer()
    
    let melGenerator:MelSpectrogram = MelSpectrogram(sampleCount: kWhisperNumSamplesInChunk, hopCount: kWhisperHopLength, melCount: kWhisperNumMels, numFFT: kWhisperNumFFTs)
    
    // These are variables which cache our current session, tasks and option
    var sessionOptions:WhisperOptions!
    private var sessionAccruedAudioSamples:[Int16] = []
    private var sessionNumAccruedAudioSamples:Int = 0
    private var sessionTranscription:[String] = []

    private var sessionSegments:[WhisperSegment] = []
    
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
        
        let unsafeAudioBufferList = UnsafeMutableAudioBufferListPointer(&audioBufferList)
        
        for (buffer) in unsafeAudioBufferList
        {
            let audioSampleArray:[Int16] = buffer.convertInt16()
                
            let samplesWeNeedToAccrueForAProperChunk = audioSampleArray[0 ... samplesToAccrue - 1]
            
            self.sessionAccruedAudioSamples.insert(contentsOf: samplesWeNeedToAccrueForAProperChunk, at: self.sessionNumAccruedAudioSamples)
                
            self.sessionNumAccruedAudioSamples = self.sessionNumAccruedAudioSamples + samplesWeNeedToAccrueForAProperChunk.count
            
            if (self.sessionAccruedAudioSamples.count == Whisper.kWhisperNumSamplesInChunk)
            {
                self.mainDeccodeLogicFromTranscribe(audio: self.sessionAccruedAudioSamples)
                
                self.sessionAccruedAudioSamples = []
                self.sessionNumAccruedAudioSamples = 0
            }
            
            // Accrue whatever remaining Samples in our current samples buffer we have..
            if (remainingCurrentSamplesInBuffer > 0)
            {
                let numSamplesWeHaveAccruedFromThisSampleBuffer = samplesWeNeedToAccrueForAProperChunk.count - 1
                
                let remainingSampleCount = Whisper.kWhisperNumSamplesInChunk - self.sessionNumAccruedAudioSamples
                
                let samplesToAccrue = min(remainingCurrentSamplesInBuffer, remainingSampleCount);

                let remainingSamplesWeNeedToAccrueForAProperChunk = audioSampleArray[numSamplesWeHaveAccruedFromThisSampleBuffer ... (numSamplesWeHaveAccruedFromThisSampleBuffer + samplesToAccrue - 1)]

                self.sessionAccruedAudioSamples.insert(contentsOf: remainingSamplesWeNeedToAccrueForAProperChunk, at: self.sessionNumAccruedAudioSamples)
                self.sessionNumAccruedAudioSamples = self.sessionNumAccruedAudioSamples + remainingSamplesWeNeedToAccrueForAProperChunk.count
            }
        }
        
        // TODO:
        // We might have residual audio samples that dont quite fill a full audio chunk (ie num frames is not equal to Whisper.kWhisperNumSamplesInChunk
        // Handle that here.
        
        // ....
    }
        
    func transcribe(assetURL:URL, options:WhisperOptions) async -> String
    {
        self.startWhisperSession(options: options)
        
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
    
    private func encode(audio: [Int16]) throws -> MLShapedArray<Float> {
        // TODO: Fix our vDSP based mel processor

        let mel:[Float] = melGenerator.processData(audio: audio)
        let melRosa:[Float] = melGenerator.processDataRosa(audio: audio)
        let melPreProcessed = MelSpectrogram.loadReferencePythonRawMelToDebugShit()
        
        self.saveNormalizedMelToDisk(mel: mel, url: URL(fileURLWithPath: "/Users/vade/Downloads/rawMel-normalized.raw"))
        self.saveNormalizedMelToDisk(mel: melRosa, url: URL(fileURLWithPath: "/Users/vade/Downloads/rawMel-rosa-normalized.raw"))
        self.saveNormalizedMelToDisk(mel: melPreProcessed, url: URL(fileURLWithPath: "/Users/vade/Downloads/rawMel-python-normalized.raw"))

        let array = MLShapedArray(scalars: mel, shape: [1, 80, 3000])
        
        let encoded = try encoderModel.prediction(logmel_data:array).var_719ShapedArray
        return encoded
//        return array
    }
    
 
    // https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L102
    private func decodeWithFallback(audio:[Int16]) -> WhisperDecodingResult?
    {
        do {
            let audioFeatures = try self.encode(audio: audio)
            
            var decodingOptions = WhisperDecodingOptions(task: self.sessionOptions.task)
            
            var decodeResult:WhisperDecodingResult? = nil
            
            for (t) in self.sessionOptions.temperatureSchedule
            {
                // Current pass decoding options
                if ( t > 0.0)
                {
                    // disable beam_size and patience when t > 0
                    decodingOptions.beamSize = nil
                    decodingOptions.patience = nil
                }
                else
                {
                    decodingOptions.bestOf = nil
                }
                
                // Set the current temperature from our temperature schedule
                decodingOptions.temperature = t
                
                decodeResult = try self.decode(audioFeatures: audioFeatures,
                                               decodingOptions: decodingOptions)
                
                var needsFallback = false
                
                if let decodeResult = decodeResult
                {
                    if decodeResult.compressionRatio > self.sessionOptions.compressionRatioTresh
                    {
                        needsFallback = true
                    }
                    
                    if (decodeResult.avgLogProbs < self.sessionOptions.logProbThresh)
                    {
                        needsFallback = true
                    }
                }
                
                if ( needsFallback == false)
                {
                    return decodeResult
                }
            }
            
            return decodeResult
        }
        catch let error
        {
            print("Unable to process audio frames", error)
            return nil
        }
    }
    
    // See https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L616
    private func decode(audioFeatures: MLShapedArray<Float>, decodingOptions:WhisperDecodingOptions) throws -> Whisper.WhisperDecodingResult {
        
        // SOT Initialize sequence
        var tokens:[Int] = []
        var timestampTokens:[Int] = []

        // create sot sequence - multilingual model always needs a task and
        // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L325
        // https://github.com/huggingface/transformers/blob/main/tests/models/whisper/test_tokenization_whisper.py
        tokens.append(WhisperTokenizer.sotToken)
        tokens.append(WhisperTokenizer.langToken)
        tokens.append(WhisperTokenizer.transcribeToken)
        
        // No Time Stamps
        if ( self.sessionOptions.format == WhisperTranscriptFormat.Text )
        {
            tokens.append(WhisperTokenizer.notToken)
        }
        
        var nextToken = 0
        var nextTSToken = WhisperTokenizer.begToken

        // More or less main loop https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L584
        while ( nextToken != WhisperTokenizer.eotToken )
        {
            autoreleasepool {
 
                let tokensArray = self.tokenizer.tokensToMultiArray(tokens)

                let decoded = try! decoderModel.prediction(token_data: tokensArray, audio_data: audioFeatures).var_1131

                let (textToken, tsToken) = self.tokenizer.nextTokenGreedy(decoded: decoded)

                nextToken = textToken
                nextTSToken = tsToken

                timestampTokens.append(nextTSToken)
                tokens.append(nextToken)

                // Verbose debugging as we iterate
//                let transcription = self.tokenizer.decode(tokens: tokens)//
//                print(transcription)
            }
        }

        // TODO: Implement calculation of other decodingResult requirements
        var decodingResult = WhisperDecodingResult(tokens: tokens, text: self.tokenizer.decode(tokens: tokens))
        
            
        return decodingResult
    }
    
    // See https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L199
    // Beam or Greedy sampling logic goes here
    private func decodeTokenUpdate(decodeOptions:WhisperDecodingOptions, tokens:[Int], logits:MLShapedArray<Float>, sumLogProbs:[Int]) throws -> (tokens:[Int], completed:Bool)
    {
        switch (decodeOptions.decodingStetegy)
        {
        case .Greedy:
            throw WhisperError.notImplementedYet
//            return self.decodeGreedyStrategy(decodeOptions:decodeOptions, tokens: tokens, logits: logits, sumLogProbs: sumLogProbs)
            
        case .BeamSearch:
            throw WhisperError.notImplementedYet
        }
    }
    
    // See "Greedy Decoder"
    // https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L249
//    private func decodeGreedyStrategy(decodeOptions:WhisperDecodingOptions, tokens:[Int], logits:MLMultiArray, sumLogProbs:[Int]) -> (tokens:[Int], completed:Bool)
//    {
//        let temp = decodeOptions.temperature
//
//        if (temp == 0)
//        {
//            let next_tokens =
//        }
//    }

    // See BeamSearch
    // https://github.com/openai/whisper/blob/12e1089462a2ea92e9ade6145e7be5b6883676ff/whisper/decoding.py#L277
//    private func decodeBeamSearchStrategy(tdecodeOptions:WhisperDecodingOptions, okens:[Int], logits:MLMultiArray, sumLogProbs:[Int]) -> (tokens:[Int], completed:Bool)
//    {
//
//    }

    // https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L175
    private func mainDeccodeLogicFromTranscribe(audio:[Int16])
    {
        
        // Timestamp shit

        // Today, we dont support audio frames other than the full 3000 Mel frame count
        // So seek is count of a mel chunk (3000) * hop length / sample rate
        // See : for reference https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L176
        let idForSegment = self.sessionSegments.count
        let segmentFrameCount = Whisper.kWhisperNumSamplesInMel
        let seek = self.sessionSegments.count * segmentFrameCount
        
        let timestampOffset = Float64(seek * Whisper.kWhisperHopLength / Whisper.kWhisperSampleRate)
        let segmentDuration =  Float64(segmentFrameCount * Whisper.kWhisperHopLength / Whisper.kWhisperSampleRate)
        
        print("segment start, segment duration", timestampOffset, segmentDuration)

        if let result:WhisperDecodingResult = self.decodeWithFallback(audio: audio)
        {
            if ( self.sessionOptions.verbose )
            {
                print (result.text)
            }
            
            let currentSegment = Whisper.WhisperSegment(id: idForSegment,
                                                        seek: seek,
                                                        startTime: CMTimeMakeWithSeconds(timestampOffset, preferredTimescale: 600),
                                                        endTime: CMTimeMakeWithSeconds(segmentDuration, preferredTimescale: 600),
                                                        textTokens: result.tokens,
                                                        decodedText: result.text)
            
            self.sessionSegments.append(currentSegment)
            
        }

    }
    
    private func resetState()
    {
        // Reset our state
        self.sessionSegments = []
        self.sessionAccruedAudioSamples = []
        self.sessionNumAccruedAudioSamples = 0
    }
    
    // MARK: Debug Methods

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
    
    private func saveNormalizedMelToDisk(mel:[Float], url:URL)
    {
        let normalizedFloatMel =  self.normalize(array: mel )
        
        normalizedFloatMel.withUnsafeBufferPointer { unsafeMel in
            
            let data = Data(buffer: unsafeMel)
            do {
                try data.write(to: url)
            }
            catch {
            }
        }
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
