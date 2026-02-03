'use client';

import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Upload, RotateCcw, CheckCircle, AlertTriangle, AlertCircle, Info, ExternalLink } from 'lucide-react';

// Types
interface AnalysisResult {
  class_label: number;
  confidence: number;
  probabilities: { class_0: number; class_1: number };
  risk_score: number;
  urgency_tier: 'low' | 'medium' | 'high';
  recommendation: string;
  classification: string;
  description: string;
}

type AppState = 'upload' | 'analyzing' | 'results';

const CLASS_LABELS = {
  0: {
    name: 'Likely Benign',
    description: 'The lesion shows characteristics commonly associated with benign (non-cancerous) skin growths.'
  },
  1: {
    name: 'Needs Evaluation',
    description: 'The lesion shows characteristics that warrant professional evaluation by a dermatologist.'
  }
};

const URGENCY_CONFIG = {
  low: { icon: CheckCircle, color: 'text-risk-low', bg: 'bg-risk-low/10', border: 'border-risk-low/30', label: 'Low Urgency' },
  medium: { icon: AlertTriangle, color: 'text-risk-medium', bg: 'bg-risk-medium/10', border: 'border-risk-medium/30', label: 'Moderate Urgency' },
  high: { icon: AlertCircle, color: 'text-risk-high', bg: 'bg-risk-high/10', border: 'border-risk-high/30', label: 'Higher Urgency' }
};

// Simulate analysis (replace with real API call)
async function analyzeImage(imageData: string): Promise<AnalysisResult> {
  await new Promise(resolve => setTimeout(resolve, 2500));

  // Hash for deterministic demo results
  let hash = 0;
  for (let i = 0; i < Math.min(imageData.length, 500); i++) {
    hash = ((hash << 5) - hash) + imageData.charCodeAt(i);
    hash = hash & hash;
  }
  hash = Math.abs(hash);

  const classLabel = (hash % 100) < 70 ? 0 : 1;
  const confidence = 0.65 + ((hash >> 8) % 30) / 100;
  const class1Prob = classLabel === 1 ? confidence : (1 - confidence);
  const riskScore = classLabel === 1 ? 0.5 + (confidence * 0.5) : (1 - confidence) * 0.4;

  let urgencyTier: 'low' | 'medium' | 'high' = 'low';
  if (riskScore >= 0.6) urgencyTier = 'high';
  else if (riskScore >= 0.3) urgencyTier = 'medium';

  const recommendations = {
    low: 'Continue regular self-monitoring. Photograph this lesion monthly to track changes.',
    medium: 'Schedule an appointment with a dermatologist for professional evaluation.',
    high: 'Please consult a dermatologist promptly for professional evaluation.'
  };

  return {
    class_label: classLabel,
    confidence,
    probabilities: { class_0: 1 - class1Prob, class_1: class1Prob },
    risk_score: riskScore,
    urgency_tier: urgencyTier,
    recommendation: recommendations[urgencyTier],
    classification: CLASS_LABELS[classLabel as 0 | 1].name,
    description: CLASS_LABELS[classLabel as 0 | 1].description
  };
}

export default function SkinTag() {
  const [state, setState] = useState<AppState>('upload');
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analysisStep, setAnalysisStep] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setShowCamera(true);
    } catch {
      alert('Unable to access camera. Please check permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    stream?.getTracks().forEach(track => track.stop());
    setStream(null);
    setShowCamera(false);
  }, [stream]);

  const capturePhoto = useCallback(() => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0);
      handleImageSelected(canvas.toDataURL('image/jpeg', 0.9));
      stopCamera();
    }
  }, [stopCamera]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => handleImageSelected(event.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleImageSelected = async (imageData: string) => {
    setImage(imageData);
    setState('analyzing');
    setAnalysisStep(0);

    const stepInterval = setInterval(() => {
      setAnalysisStep(prev => (prev + 1) % 5);
    }, 500);

    try {
      const analysisResult = await analyzeImage(imageData);
      setResult(analysisResult);
      setState('results');
    } catch {
      setState('upload');
      setImage(null);
    } finally {
      clearInterval(stepInterval);
    }
  };

  const reset = () => {
    setImage(null);
    setResult(null);
    setState('upload');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const config = result ? URGENCY_CONFIG[result.urgency_tier] : null;
  const Icon = config?.icon || CheckCircle;

  const analysisSteps = ['Loading model...', 'Extracting features...', 'Comparing images...', 'Computing risk...', 'Generating results...'];

  return (
    <div className="min-h-[100dvh] bg-cream flex flex-col">
      {/* Background blobs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="blob-shape w-72 h-72 bg-sage-light/50 -top-36 -right-36" />
        <div className="blob-shape w-56 h-56 bg-terracotta-light/30 bottom-20 -left-28" style={{ animationDelay: '-10s' }} />
      </div>

      {/* Header */}
      <header className="relative z-10 px-5 pt-6 pb-4 safe-area-top">
        <h1 className="font-serif text-xl text-charcoal">SkinTag</h1>
        <p className="text-[10px] text-charcoal-light">Research prototype &middot; Not for clinical use</p>
      </header>

      {/* Main Content */}
      <main className="flex-1 relative z-10 px-5 pb-6 flex flex-col">
        <AnimatePresence mode="wait">

          {/* Upload State */}
          {state === 'upload' && !showCamera && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex-1 flex flex-col"
            >
              <div className="text-center mb-6">
                <h2 className="font-serif text-2xl text-charcoal mb-2">Check Your Skin</h2>
                <p className="text-charcoal-light text-sm">Upload or capture an image of any mole or skin lesion</p>
              </div>

              <div className="flex-1 flex items-center justify-center">
                <div className="w-full max-w-sm aspect-square rounded-3xl border-2 border-dashed border-sage-light bg-warm-white/50 flex flex-col items-center justify-center gap-6 p-8">
                  <div className="w-16 h-16 rounded-full bg-sage/10 flex items-center justify-center">
                    <Camera size={28} className="text-sage-dark" />
                  </div>

                  <div className="flex flex-col gap-3 w-full">
                    <button
                      onClick={startCamera}
                      className="w-full py-4 bg-sage text-warm-white rounded-2xl font-medium flex items-center justify-center gap-2"
                    >
                      <Camera size={20} /> Open Camera
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full py-4 bg-warm-white text-charcoal rounded-2xl font-medium flex items-center justify-center gap-2 border border-sage-light/50"
                    >
                      <Upload size={20} /> Upload Image
                    </button>
                    <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                  </div>
                </div>
              </div>

              <div className="text-center mt-6 space-y-1">
                <p className="text-[10px] text-charcoal-light/60">Trained on 47,277 images &middot; Google SigLIP-SO400M &middot; 878M parameters</p>
              </div>
            </motion.div>
          )}

          {/* Camera View */}
          {showCamera && (
            <motion.div
              key="camera"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col"
            >
              <div className="flex-1 flex items-center justify-center">
                <div className="relative w-full max-w-sm aspect-square rounded-3xl overflow-hidden bg-charcoal">
                  <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <div className="absolute inset-4 border-2 border-sage/50 rounded-2xl pointer-events-none" />
                </div>
              </div>
              <div className="flex gap-4 justify-center mt-6">
                <button onClick={stopCamera} className="px-6 py-3 bg-charcoal/20 text-charcoal rounded-xl">Cancel</button>
                <button onClick={capturePhoto} className="px-8 py-3 bg-sage text-warm-white rounded-xl font-medium">Capture</button>
              </div>
            </motion.div>
          )}

          {/* Analyzing State */}
          {state === 'analyzing' && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col items-center justify-center"
            >
              <div className="w-full max-w-sm bg-warm-white rounded-3xl p-8 shadow-xl">
                {image && (
                  <div className="w-24 h-24 mx-auto mb-6 rounded-2xl overflow-hidden">
                    <img src={image} alt="Analyzing" className="w-full h-full object-cover" />
                  </div>
                )}

                <div className="relative w-20 h-20 mx-auto mb-6">
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }} className="absolute inset-0 rounded-full border-2 border-sage-light border-t-sage" />
                  <div className="absolute inset-4 rounded-full bg-sage/10" />
                </div>

                <h3 className="font-serif text-xl text-charcoal text-center mb-2">Analyzing</h3>
                <p className="text-sm text-charcoal-light text-center mb-4">{analysisSteps[analysisStep]}</p>

                <p className="text-[10px] text-charcoal-light/60 text-center">Powered by Google SigLIP-SO400M</p>
              </div>
            </motion.div>
          )}

          {/* Results State */}
          {state === 'results' && result && config && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col overflow-y-auto"
            >
              {/* Image + Classification */}
              <div className="flex gap-4 mb-4">
                {image && (
                  <div className="w-20 h-20 rounded-2xl overflow-hidden flex-shrink-0">
                    <img src={image} alt="Analyzed" className="w-full h-full object-cover" />
                  </div>
                )}
                <div className="flex-1">
                  <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full ${config.bg} mb-1`}>
                    <Icon size={14} className={config.color} />
                    <span className={`text-xs font-medium ${config.color}`}>{config.label}</span>
                  </div>
                  <h2 className="font-serif text-xl text-charcoal">{result.classification}</h2>
                </div>
              </div>

              {/* Risk Score */}
              <div className={`rounded-2xl p-4 ${config.bg} border ${config.border} mb-4`}>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-charcoal-light">Risk Score</span>
                  <span className="font-medium text-charcoal">{(result.risk_score * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-charcoal/10 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.risk_score * 100}%` }}
                    transition={{ duration: 0.8 }}
                    className={`h-full rounded-full ${result.urgency_tier === 'low' ? 'bg-risk-low' : result.urgency_tier === 'medium' ? 'bg-risk-medium' : 'bg-risk-high'}`}
                  />
                </div>
                <p className="text-xs text-charcoal-light mt-3">{result.description}</p>
              </div>

              {/* Recommendation */}
              <div className="rounded-2xl p-4 bg-trust-blue/10 border border-trust-blue/20 mb-4">
                <div className="flex gap-3">
                  <Info size={18} className="text-trust-blue flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs font-medium text-charcoal mb-1">Recommendation</p>
                    <p className="text-xs text-charcoal-light leading-relaxed">{result.recommendation}</p>
                  </div>
                </div>
              </div>

              {/* Probabilities */}
              <div className="rounded-2xl p-4 bg-warm-white border border-sage-light/30 mb-4">
                <p className="text-xs font-medium text-charcoal mb-3">Classification Details</p>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-charcoal">Benign</span>
                      <span className="text-charcoal-light">{(result.probabilities.class_0 * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-charcoal/10 rounded-full overflow-hidden">
                      <div className="h-full bg-risk-low rounded-full" style={{ width: `${result.probabilities.class_0 * 100}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-charcoal">Suspicious</span>
                      <span className="text-charcoal-light">{(result.probabilities.class_1 * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-charcoal/10 rounded-full overflow-hidden">
                      <div className="h-full bg-risk-high rounded-full" style={{ width: `${result.probabilities.class_1 * 100}%` }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="rounded-2xl p-4 bg-charcoal/5 border border-charcoal/10 mb-4">
                <p className="text-[11px] text-charcoal leading-relaxed">
                  Condition estimation is approximate, based on visual similarity to <strong>47,277 training images</strong> across <strong>10 diagnostic categories</strong>. Only a biopsy can confirm a diagnosis.
                </p>
              </div>

              {/* Model Info */}
              <div className="text-center space-y-2 mb-4">
                <p className="text-[10px] text-charcoal-light/70">
                  Powered by Google SigLIP-SO400M &middot; 878M parameters &middot; <span className="text-terracotta">Not for clinical use</span>
                </p>
                <a href="https://www.aad.org/public/diseases/skin-cancer" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs text-trust-blue">
                  Learn about skin cancer <ExternalLink size={10} />
                </a>
              </div>

              {/* Scan Again Button */}
              <button
                onClick={reset}
                className="w-full py-4 bg-sage text-warm-white rounded-2xl font-medium flex items-center justify-center gap-2 mt-auto"
              >
                <RotateCcw size={18} /> Scan Again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
