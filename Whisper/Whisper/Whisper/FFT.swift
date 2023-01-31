//
//  FFTImplementations.swift
//  Whisper
//
//  Created by Anton Marini on 1/12/23.
//

import Foundation
import Accelerate

//figure out if i need to remove the dc offset and nyquist from the returned arrays because maybe im loosing a bin?


public class WhisperFFT
{

    var window:[Double]
    var numFFT:Int = 400

    init(numFFT:Int)
    {
        self.numFFT = numFFT
        
        self.window = vDSP.window(ofType: Double.self,
                                         usingSequence: .hanningDenormalized,
                                         count: self.numFFT,
                                         isHalfWindow: false)
    }
}

public class NumpyRFFT : WhisperFFT
{
//    var fft : vDSP.FFT<DSPDoubleSplitComplex>
    var n:Int!
    var nOver2:Int!
    var logSize:vDSP_Length!
    
    override init(numFFT: Int)
    {
        super.init(numFFT:numFFT)

        self.logSize = vDSP_Length(floor(log2(Float(self.numFFT))))
        self.n = numFFT
        self.nOver2 = (numFFT / 2)

//        self.fft = vDSP.FFT(log2n: UInt(self.nOver2),
//                           radix: .radix2,
//                           ofType: DSPDoubleSplitComplex.self)!

    }

    
    public func forward(_ audioFrame:[Double]) -> ([Double], [Double])
    {
        var windowedAudioFrame = [Double](repeating: 0, count: self.numFFT)
        
        vDSP.multiply(audioFrame,
                      self.window,
                      result: &windowedAudioFrame)
    
        var sampleReal:[Double] = [Double](repeating: 0, count: self.nOver2)
        var sampleImaginary:[Double] = [Double](repeating: 0, count: self.nOver2)

        var resultReal:[Double] = [Double](repeating: 0, count: n)
        var resultImaginary:[Double] = [Double](repeating: 0, count:n)

        let fftSetup:FFTSetupD = vDSP_create_fftsetupD(self.logSize, FFTRadix(kFFTRadix2))!;

        
        sampleReal.withUnsafeMutableBytes { unsafeReal in
            sampleImaginary.withUnsafeMutableBytes { unsafeImaginary in
                
                resultReal.withUnsafeMutableBytes { unsafeResultReal in
                    resultImaginary.withUnsafeMutableBytes { unsafeResultImaginary in
                        
                        var complexSignal = DSPDoubleSplitComplex(realp: unsafeReal.bindMemory(to: Double.self).baseAddress!,
                                                                  imagp: unsafeImaginary.bindMemory(to: Double.self).baseAddress!)
                        
                        let complexResult = DSPDoubleSplitComplex(realp: unsafeResultReal.bindMemory(to: Double.self).baseAddress!,
                                                                  imagp: unsafeResultImaginary.bindMemory(to: Double.self).baseAddress!)

                        // Treat our windowed audio as a Interleaved Complex
                        // And convert it into a split complex Signal
                        windowedAudioFrame.withUnsafeBytes { unsafeAudioBytes in
                            let letInterleavedComplexAudio = [DSPDoubleComplex](unsafeAudioBytes.bindMemory(to: DSPDoubleComplex.self))
                                                                                
                            vDSP_ctozD(letInterleavedComplexAudio, 2, &complexSignal, 1, vDSP_Length(self.nOver2)) ;

                            vDSP_fft_zripD (fftSetup, &complexSignal, 1, self.logSize, FFTDirection(kFFTDirection_Forward));

                        }
                        
                        // Scale by 1/2 : https://stackoverflow.com/questions/51804365/why-is-fft-different-in-swift-than-in-python
                        var scaleFactor = Double( 1.0/2.0 ) // * 1.165 ??
                        vDSP_vsmulD(complexSignal.realp, 1, &scaleFactor, complexSignal.realp, 1, vDSP_Length(self.nOver2))
                        vDSP_vsmulD(complexSignal.imagp, 1, &scaleFactor, complexSignal.imagp, 1, vDSP_Length(self.nOver2))
                        
                        
                        // Borrowed from https://github.com/jseales/numpy-style-fft-in-obj-c
                        complexResult.realp[0] = complexSignal.realp[0];
                        complexResult.imagp[0] = 0;
                        complexResult.realp[self.nOver2] = complexSignal.imagp[0];
                        complexResult.imagp[self.nOver2] = 0;

                        for (i) in 1 ..< self.nOver2
                        {
                            complexResult.realp[i] = complexSignal.realp[i];
                            complexResult.imagp[i] = complexSignal.imagp[i];
                            // Complex conjugate is mirrored (?)
                            complexResult.realp[n - i] = complexSignal.realp[i];
                            complexResult.imagp[n - i] = complexSignal.imagp[i];
                        }
                    }
                }
            }
        }
        
        return (resultReal, resultImaginary)
    }
    
}

public class NumpyFFT : WhisperFFT
{
//    var fft : vDSP.FFT<DSPDoubleSplitComplex>
    var n:Int!
    var nOver2:Int!
    var logSize:vDSP_Length!
    
    override init(numFFT: Int)
    {
        super.init(numFFT:numFFT)

        self.logSize = vDSP_Length(floor(log2(Float(self.numFFT))))
        self.n = numFFT
        self.nOver2 = (numFFT / 2)

//        self.fft = vDSP.FFT(log2n: UInt(self.nOver2),
//                           radix: .radix2,
//                           ofType: DSPDoubleSplitComplex.self)!

    }

    
    public func forward(_ audioFrame:[Double]) -> ([Double], [Double])
    {
        var sampleReal = [Double](repeating: 0, count: self.numFFT)
        
        vDSP.multiply(audioFrame,
                      self.window,
                      result: &sampleReal)
    
        var sampleImaginary:[Double] = [Double](repeating: 0, count: self.numFFT)

        let fftSetup:FFTSetupD = vDSP_create_fftsetupD(self.logSize, FFTRadix(kFFTRadix2))!;
        
        sampleReal.withUnsafeMutableBytes { unsafeReal in
            sampleImaginary.withUnsafeMutableBytes { unsafeImaginary in
                
                var complexSignal = DSPDoubleSplitComplex(realp: unsafeReal.bindMemory(to: Double.self).baseAddress!,
                                                          imagp: unsafeImaginary.bindMemory(to: Double.self).baseAddress!)
                
                vDSP_fft_zipD(fftSetup, &complexSignal, 1, self.logSize, FFTDirection(kFFTDirection_Forward));
            }
        }
        
        return (sampleReal, sampleImaginary)
    }
    
}
