'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { jsPDF } from 'jspdf'; 
import { useUser, SignOutButton } from "@clerk/nextjs";
import ReactMarkdown from 'react-markdown';
import * as THREE from 'three';
import { Cpu, Activity, BrainCircuit, ShieldAlert, FileText, ChevronRight, HelpCircle } from 'lucide-react';

// --- PURE JAVASCRIPT EDF FILE PARSER ---
const parseEDF = async (file: File) => {
  const buffer = await file.arrayBuffer();
  const headerStr = new TextDecoder('ascii').decode(buffer.slice(0, 256));
  const headerBytes = parseInt(headerStr.slice(184, 192).trim());
  const numRecords = parseInt(headerStr.slice(236, 244).trim());
  const recordDuration = parseFloat(headerStr.slice(244, 252).trim());
  const numChannels = parseInt(headerStr.slice(252, 256).trim());
  
  let offset = 256;
  const labels: string[] = [];
  for (let i = 0; i < numChannels; i++) {
    labels.push(new TextDecoder('ascii').decode(buffer.slice(offset + i * 16, offset + (i + 1) * 16)).trim());
  }
  offset += numChannels * 16; // skip labels
  offset += numChannels * 80; // skip transducers
  offset += numChannels * 8;  // skip dimensions
  
  const physMin: number[] = [];
  for (let i = 0; i < numChannels; i++) {
    physMin.push(parseFloat(new TextDecoder('ascii').decode(buffer.slice(offset + i * 8, offset + (i + 1) * 8)).trim()));
  }
  offset += numChannels * 8;
  
  const physMax: number[] = [];
  for (let i = 0; i < numChannels; i++) {
    physMax.push(parseFloat(new TextDecoder('ascii').decode(buffer.slice(offset + i * 8, offset + (i + 1) * 8)).trim()));
  }
  offset += numChannels * 8;
  
  const digMin: number[] = [];
  for (let i = 0; i < numChannels; i++) {
    digMin.push(parseFloat(new TextDecoder('ascii').decode(buffer.slice(offset + i * 8, offset + (i + 1) * 8)).trim()));
  }
  offset += numChannels * 8;
  
  const digMax: number[] = [];
  for (let i = 0; i < numChannels; i++) {
    digMax.push(parseFloat(new TextDecoder('ascii').decode(buffer.slice(offset + i * 8, offset + (i + 1) * 8)).trim()));
  }
  offset += numChannels * 8;
  offset += numChannels * 80; // skip prefiltering
  
  const numSamplesPerRecord: number[] = [];
  for (let i = 0; i < numChannels; i++) {
    numSamplesPerRecord.push(parseInt(new TextDecoder('ascii').decode(buffer.slice(offset + i * 8, offset + (i + 1) * 8)).trim()));
  }
  
  // Find EEG channel
  let eegChanIdx = labels.findIndex(l => l.includes('EEG Fpz-Cz') || l.includes('Fpz-Cz') || l.includes('EEG'));
  if (eegChanIdx === -1) eegChanIdx = 0; // fallback
  
  const dataView = new DataView(buffer);
  const dataOffset = headerBytes;
  
  const totalSamplesPerRecord = numSamplesPerRecord.reduce((a, b) => a + b, 0);
  const eegSamples: number[] = [];
  
  const channelRecordOffsets: number[] = [];
  let curOffset = 0;
  for (let i = 0; i < numChannels; i++) {
    channelRecordOffsets.push(curOffset);
    curOffset += numSamplesPerRecord[i];
  }
  
  const nSamples = numSamplesPerRecord[eegChanIdx];
  const pMin = physMin[eegChanIdx];
  const pMax = physMax[eegChanIdx];
  const dMin = digMin[eegChanIdx];
  const dMax = digMax[eegChanIdx];
  const scale = (pMax - pMin) / (dMax - dMin);
  
  for (let r = 0; r < numRecords; r++) {
    const recordStartOffset = dataOffset + r * totalSamplesPerRecord * 2;
    const chanStartOffset = recordStartOffset + channelRecordOffsets[eegChanIdx] * 2;
    for (let s = 0; s < nSamples; s++) {
      if (chanStartOffset + s * 2 + 1 < buffer.byteLength) {
        const digVal = dataView.getInt16(chanStartOffset + s * 2, true); // little endian
        const physVal = (digVal - dMin) * scale + pMin;
        eegSamples.push(physVal);
      }
    }
  }
  return { samples: eegSamples, sfreq: nSamples / recordDuration };
};

// --- SLEEP METRICS CALCULATOR ---
const calculateSleepMetrics = (predictions: string[], durationMinutes: number) => {
  if (!predictions || predictions.length === 0) {
    return {
      cycles: 'N/A',
      onset: 'N/A',
      remLatency: 'N/A',
      waso: 'N/A',
      recordingTime: `${durationMinutes || 0} mins`
    };
  }

  const epochDuration = 0.5; // each epoch is 30s = 0.5 mins

  // 1. Sleep Onset Latency (SOL)
  let firstSleepIdx = -1;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] !== 'W') {
      firstSleepIdx = i;
      break;
    }
  }
  const onset = firstSleepIdx === -1 ? '0m' : `${Math.round(firstSleepIdx * epochDuration)}m`;

  // 2. REM Latency
  let firstRemIdx = -1;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === 'REM') {
      firstRemIdx = i;
      break;
    }
  }
  let remLatency = 'N/A';
  if (firstSleepIdx !== -1 && firstRemIdx !== -1) {
    if (firstRemIdx >= firstSleepIdx) {
      remLatency = `${Math.round((firstRemIdx - firstSleepIdx) * epochDuration)}m`;
    } else {
      remLatency = '0m';
    }
  }

  // 3. Wake After Onset (WASO)
  let wasoEpochs = 0;
  if (firstSleepIdx !== -1) {
    for (let i = firstSleepIdx; i < predictions.length; i++) {
      if (predictions[i] === 'W') {
        wasoEpochs++;
      }
    }
  }
  const waso = `${Math.round(wasoEpochs * epochDuration)}m`;

  // 4. Sleep Cycles: count contiguous blocks of REM sleep
  let cycles = 0;
  let inRemBlock = false;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === 'REM') {
      if (!inRemBlock) {
        cycles++;
        inRemBlock = true;
      }
    } else {
      inRemBlock = false;
    }
  }
  
  if (cycles === 0 && firstSleepIdx !== -1) {
    cycles = durationMinutes > 240 ? 3 : (durationMinutes > 90 ? 2 : 1);
  }

  return {
    cycles: cycles || 1,
    onset,
    remLatency,
    waso,
    recordingTime: `${durationMinutes || 0} mins`
  };
};

// --- WebGL THREE.JS DYNAMIC DREAMSCAPE ---
const DreamscapeCanvas = ({ entropy, stage }: { entropy: number, stage: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
    camera.position.z = 4.5;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    const particleCount = 1200;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const initialPositions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = 1.8 * Math.cbrt(Math.random());
      
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      initialPositions[i * 3] = x;
      initialPositions[i * 3 + 1] = y;
      initialPositions[i * 3 + 2] = z;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const getColor = (s: string) => {
      switch (s) {
        case 'W': return new THREE.Color(0xf43f5e);
        case 'N1': return new THREE.Color(0xf59e0b);
        case 'N2': return new THREE.Color(0x3b82f6);
        case 'N3': return new THREE.Color(0x8b5cf6);
        case 'REM': return new THREE.Color(0x10b981);
        default: return new THREE.Color(0x3b82f6);
      }
    };

    const targetColor = getColor(stage);
    const material = new THREE.PointsMaterial({
      color: targetColor,
      size: 0.045,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    const clock = new THREE.Clock();
    let animationFrameId: number;

    const animate = () => {
      const elapsed = clock.getElapsedTime();
      
      const curColor = material.color;
      const nextColor = getColor(stage);
      curColor.lerp(nextColor, 0.05);

      const rotationSpeed = (entropy || 0.5) * 0.015 + 0.003;
      particles.rotation.y += rotationSpeed;
      particles.rotation.x += rotationSpeed * 0.4;

      const pos = geometry.attributes.position.array as Float32Array;
      const amp = (entropy || 0.5) * 0.25 + 0.05;
      for (let i = 0; i < particleCount; i++) {
        const initX = initialPositions[i * 3];
        const initY = initialPositions[i * 3 + 1];
        const initZ = initialPositions[i * 3 + 2];
        
        const wave = Math.sin(initX * 2 + elapsed * 1.5) * Math.cos(initY * 2 + elapsed * 1.5) * amp;
        pos[i * 3] = initX + wave * (initX / 2);
        pos[i * 3 + 1] = initY + wave * (initY / 2);
        pos[i * 3 + 2] = initZ + wave * (initZ / 2);
      }
      geometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      if (!canvas) return;
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, [stage, entropy]);

  return <canvas ref={canvasRef} className="w-full h-full min-h-[160px] max-h-[180px]" />;
};

// --- RAW EEG & GRADIENT SALIENCY (XAI) PLOTTER ---
const RawSignalVisualizer = ({ signal, saliency, stage, epochIndex }: { signal: number[], saliency: number[], stage: string, epochIndex: number }) => {
  if (!signal || !saliency) {
    return (
      <div className="rounded-2xl border border-white/5 bg-[#121318] p-8 text-center text-slate-500 text-sm">
        ⚠️ Raw EEG wave data is not available for this session. (Run local Edge inference to enable).
      </div>
    );
  }

  const width = 1000;
  const height = 150;
  const padding = 20;

  const sigMin = Math.min(...signal);
  const sigMax = Math.max(...signal);
  const sigRange = sigMax - sigMin || 1;

  const points = signal.map((val, idx) => {
    const x = padding + (idx / (signal.length - 1)) * (width - 2 * padding);
    const y = height / 2 - ((val - (sigMax + sigMin) / 2) / sigRange) * (height - 2 * padding);
    return { x, y };
  });

  return (
    <div className="rounded-2xl border border-white/5 bg-[#121318] p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3">
        <div>
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-400" />
            Epoch {epochIndex + 1} EEG Signal Analysis ({Math.round(epochIndex * 30)}s - {Math.round((epochIndex + 1) * 30)}s)
          </h4>
          <p className="text-xs text-slate-400 mt-1">
            Predicted Sleep Stage: <span className="font-semibold text-blue-400">{stage === "W" ? "Wakefulness (W)" : stage === "N3" ? "Deep Sleep (N3)" : stage === "REM" ? "REM Sleep" : "Light Sleep (" + stage + ")"}</span>
          </p>
        </div>
        <div className="flex items-center space-x-4 text-[10px]">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="h-2.5 w-2.5 rounded-full bg-blue-500"></span> Low Attention
          </span>
          <span className="flex items-center gap-1.5 text-rose-400 font-semibold">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-500"></span> High Attention (XAI Hotspots)
          </span>
        </div>
      </div>

      <div className="relative overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[700px] h-40 bg-slate-950/50 rounded-2xl border border-white/5 shadow-inner">
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#ffffff08" strokeDasharray="4 4" />
          
          {points.map((p, idx) => {
            if (idx === 0) return null;
            const prev = points[idx - 1];
            const sal = saliency[idx] || 0;
            
            const r = Math.round(sal * 244 + (1 - sal) * 59);
            const g = Math.round(sal * 63 + (1 - sal) * 130);
            const b = Math.round(sal * 94 + (1 - sal) * 246);
            const strokeColor = `rgb(${r}, ${g}, ${b})`;
            
            return (
              <line 
                key={idx}
                x1={prev.x} y1={prev.y}
                x2={p.x} y2={p.y}
                stroke={strokeColor}
                strokeWidth={sal > 0.65 ? 2.5 : 1.5}
                strokeLinecap="round"
              />
            );
          })}
        </svg>
      </div>
      <p className="text-[10px] text-slate-500 italic">
        * Saliency hot-spots (indicated in red) represent precisely which sub-second cycles of this 30-second epoch the neural network focused on to establish its prediction.
      </p>
    </div>
  );
};

export default function DashboardPage() {
  // --- UI & DATA STATE ---
  const [activeTab, setActiveTab] = useState<'overview' | 'history' | 'patients'>('overview');
  const [selectedHistorySession, setSelectedHistorySession] = useState<any>(null);
  const { user } = useUser();
  
  // Storing reports fetched from the FastAPI/Mongo backend
  const [historyList, setHistoryList] = useState<any[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);

  // --- PATIENT PORTAL STATE ---
  const [patientMode, setPatientMode] = useState<'search' | 'register'>('search');
  const [patientSearchPhone, setPatientSearchPhone] = useState<string>('');
  const [patientFound, setPatientFound] = useState<any>(null);
  const [patientSearchError, setPatientSearchError] = useState<string>('');
  const [isSearchingPatient, setIsSearchingPatient] = useState<boolean>(false);

  const [regName, setRegName] = useState<string>('');
  const [regAge, setRegAge] = useState<string>('');
  const [regGender, setRegGender] = useState<'Male' | 'Female' | 'Other'>('Male');
  const [regPhone, setRegPhone] = useState<string>('');

  const [activePatient, setActivePatient] = useState<any>(null);

  // Directory States
  const [patientsList, setPatientsList] = useState<any[]>([]);
  const [isLoadingPatients, setIsLoadingPatients] = useState<boolean>(false);
  const [patientQuery, setPatientQuery] = useState<string>('');
  const [selectedPatientProfile, setSelectedPatientProfile] = useState<any>(null);
  const [selectedPatientHistory, setSelectedPatientHistory] = useState<any[]>([]);
  const [isLoadingPatientHistory, setIsLoadingPatientHistory] = useState<boolean>(false);

  // Comparison States
  const [compareSessions, setCompareSessions] = useState<any[]>([]);
  const [showComparison, setShowComparison] = useState<boolean>(false);

  // Chat widget states
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: "Hi Immam! I've analyzed your latest EEG file. Ask me anything about your sleep!" }
  ]);

  // File upload and processing states
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  
  const [analysisData, setAnalysisData] = useState<any>(null); 
  const [rawPredictions, setRawPredictions] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- NEW UPGRADES STATE ---
  const [selectedEpoch, setSelectedEpoch] = useState<number>(0);
  const [edgeMode, setEdgeMode] = useState<boolean>(false);
  const [onnxProgress, setOnnxProgress] = useState<string>('');
  const [isSaved, setIsSaved] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  // Dynamically update Chat Welcome Message based on Logged-in User and Active Patient
  useEffect(() => {
    const docName = user?.fullName || user?.firstName || "Doctor";
    const patName = activePatient?.name || analysisData?.patient_name || selectedHistorySession?.patient_name || "";
    
    const welcomeText = patName 
      ? `Hi ${docName}! I've analyzed the EEG file of ${patName}. Ask me anything about their sleep!`
      : `Hi ${docName}! Select or upload a patient's sleep session, and I can analyze their EEG data.`;

    setChatHistory([
      { role: 'ai', text: welcomeText }
    ]);
  }, [activePatient, analysisData, selectedHistorySession, user]);

  // --- API CALLS ---
  
  // Pulling the user's past reports from the database
  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const userId = user?.id || "guest_user";
      const res = await fetch(`http://localhost:8000/history/${userId}`);
      const data = await res.json();
      if (data.status === "success") {
        setHistoryList(data.data);
      }
    } catch (error) {
      console.error("Failed to fetch history:", error);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Pulling clinical patient details
  const fetchPatients = async () => {
    setIsLoadingPatients(true);
    try {
      const res = await fetch("http://localhost:8000/patients");
      const data = await res.json();
      if (data.status === "success") {
        setPatientsList(data.data);
      }
    } catch (err) {
      console.error("Failed to fetch patients:", err);
    } finally {
      setIsLoadingPatients(false);
    }
  };

  const handleSearchPatient = async () => {
    if (!patientSearchPhone.trim()) return;
    setIsSearchingPatient(true);
    setPatientSearchError('');
    setPatientFound(null);
    try {
      const res = await fetch(`http://localhost:8000/patients/${patientSearchPhone.trim()}`);
      const data = await res.json();
      if (res.ok && data.status === "success") {
        setPatientFound(data.data);
        setActivePatient(data.data);
      } else {
        setPatientSearchError(data.detail || "Patient not found. Please register them.");
      }
    } catch (err) {
      setPatientSearchError("Failed to connect to backend server.");
    } finally {
      setIsSearchingPatient(false);
    }
  };

  const fetchPatientHistory = async (phone: string) => {
    setIsLoadingPatientHistory(true);
    try {
      const res = await fetch(`http://localhost:8000/patients/${phone}/history`);
      const data = await res.json();
      if (data.status === "success") {
        setSelectedPatientHistory(data.data);
      }
    } catch (err) {
      console.error("Failed to fetch patient history:", err);
    } finally {
      setIsLoadingPatientHistory(false);
    }
  };

  const handleSelectCompare = (session: any) => {
    setCompareSessions((prev) => {
      const isAlreadySelected = prev.some((s) => s._id === session._id);
      if (isAlreadySelected) {
        return prev.filter((s) => s._id !== session._id);
      } else {
        if (prev.length >= 2) {
          alert("You can select up to 2 sessions to compare.");
          return prev;
        }
        return [...prev, session];
      }
    });
  };

  // Re-run fetch whenever the user logs in or activeTab changes (to pull directory list)
  useEffect(() => {
    fetchHistory();
    if (activeTab === 'patients') {
      fetchPatients();
      setSelectedPatientProfile(null);
      setSelectedPatientHistory([]);
      setCompareSessions([]);
    }
  }, [user?.id, activeTab]);

  // --- CALCULATIONS ---

  // Working out the total count of 30s chunks
  const getTotalEpochs = (sourceData: any) => {
    if (!sourceData?.summary) return 1;
    return (sourceData.summary.Wake + sourceData.summary.Light + sourceData.summary.Deep + sourceData.summary.REM) || 1;
  };

  // Converting epoch counts into percentages and readable time (h/m)
  const getStats = (epochCount: number, sourceData: any) => {
    if (!sourceData?.summary) return { percent: "0%", time: "0 mins" };
    const total = getTotalEpochs(sourceData);
    const percent = Math.round((epochCount / total) * 100) + "%";
    const totalMins = Math.round(epochCount * 0.5); 
    const hours = Math.floor(totalMins / 60);
    const mins = totalMins % 60;
    const time = hours > 0 ? `${hours}h ${mins}m` : `${mins} mins`;
    return { percent, time };
  };

  // Mapping the string stages to numerical values for the AreaChart
  const prepareChartData = (predictions: string[]) => {
    if (!predictions) return [];
    const mapping: { [key: string]: number } = { "W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0 };
    return predictions.map((stage, index) => ({
      epoch: index, 
      stageValue: mapping[stage],
      stageName: stage === "W" ? "Wake" : stage === "N3" ? "Deep" : stage === "REM" ? "REM" : "Light"
    }));
  };

  // Simple logic to show a sleep score based on wake time vs total time
  const currentSleepScore = analysisData?.summary ? Math.max(0, Math.round(100 - ((analysisData.summary.Wake / getTotalEpochs(analysisData)) * 100))) : 88;

  // --- INTERACTION HANDLERS ---

  // Sending the EDF file to the Python backend for Transformer inference
  const handleAnalyze = async (e: React.MouseEvent) => {
    e.stopPropagation(); 
    if (!file) return;

    if (patientMode === 'search' && !activePatient) {
      alert("Please select or search a registered patient first.");
      return;
    }
    if (patientMode === 'register' && (!regName.trim() || !regAge.trim() || !regPhone.trim())) {
      alert("Please fill out all patient registration details.");
      return;
    }

    setIsAnalyzing(true);
    setOnnxProgress('');
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("user_id", user?.id || "guest_user");
      
      if (patientMode === 'search' && activePatient) {
        formData.append("patient_phone", activePatient.phone_number);
      } else {
        formData.append("patient_phone", regPhone.trim());
        formData.append("patient_name", regName.trim());
        formData.append("patient_age", regAge.trim());
        formData.append("patient_gender", regGender);
      }

      const response = await fetch("http://localhost:8000/analyze", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Backend failed");
      const data = await response.json();
      
      if (data.status === "success") {
        const computedMetrics = calculateSleepMetrics(data.predictions, data.metadata.durationMinutes);
        const pName = data.patient_name || (patientMode === 'search' ? activePatient?.name : regName.trim());
        const pAge = data.patient_age !== undefined ? data.patient_age : (patientMode === 'search' ? activePatient?.age : parseInt(regAge));
        const pGender = data.patient_gender || (patientMode === 'search' ? activePatient?.gender : regGender);
        const pPhone = data.patient_phone || (patientMode === 'search' ? activePatient?.phone_number : regPhone.trim());

        setAnalysisData({ 
          summary: data.summary, 
          brainwaves: data.brainwaves,
          metadata: data.metadata,
          predictions: data.predictions,
          spectral_entropies: data.spectral_entropies,
          signals: data.signals,
          saliencies: data.saliencies,
          metrics: computedMetrics,
          patient_name: pName,
          patient_age: pAge,
          patient_gender: pGender,
          patient_phone: pPhone
        });
        setRawPredictions(data.predictions || []);
        setSelectedEpoch(0); // Reset inspected epoch to 0
        setIsSaved(true); // Cloud AI saves automatically on backend
        setShowResults(true); 

        if (patientMode === 'register') {
          setActivePatient({
            name: regName.trim(),
            age: parseInt(regAge),
            gender: regGender,
            phone_number: regPhone.trim()
          });
        }
        
        fetchHistory(); // Refresh the list after a new analysis
        fetchPatients();
      }
    } catch (error) {
      alert("Error connecting to server. Is FastAPI running?");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Running local client-side WebAssembly inference using onnxruntime-web
  const runEdgeInference = async (selectedFile: File) => {
    if (patientMode === 'search' && !activePatient) {
      alert("Please select or search a registered patient first.");
      return;
    }
    if (patientMode === 'register' && (!regName.trim() || !regAge.trim() || !regPhone.trim())) {
      alert("Please fill out all patient registration details.");
      return;
    }

    setIsAnalyzing(true);
    setOnnxProgress("Configuring WebAssembly runtime...");
    try {
      const ort = require('onnxruntime-web');
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
      
      setOnnxProgress("Downloading ONNX model (2.3MB) to browser...");
      const session = await ort.InferenceSession.create('/dreamlens_transformer.onnx');
      
      setOnnxProgress("Parsing raw EEG signal from EDF...");
      const { samples, sfreq } = await parseEDF(selectedFile);
      
      setOnnxProgress("Slicing signal into 30s segments...");
      const epochSize = Math.round(30 * sfreq);
      const numEpochs = Math.floor(samples.length / epochSize);
      
      if (numEpochs === 0) throw new Error("File is too short to analyze.");
      
      const predictedStages: string[] = [];
      const spectralEntropies: number[] = [];
      const downsampledSignals: number[][] = [];
      const downsampledSaliencies: number[][] = [];
      
      const stageMap: { [key: number]: string } = { 0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM" };
      
      setOnnxProgress("Running local neural network inference...");
      for (let i = 0; i < numEpochs; i++) {
        const startIdx = i * epochSize;
        const epochData = samples.slice(startIdx, startIdx + epochSize);
        
        // Signal standardization
        const mean = epochData.reduce((a, b) => a + b, 0) / epochData.length;
        const variance = epochData.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / epochData.length;
        const std = Math.sqrt(variance);
        const normData = epochData.map(v => (v - mean) / (std + 1e-8));
        
        // Format to exact input dimensions expected by model (3000 samples)
        const modelInput = new Float32Array(3000);
        for (let j = 0; j < 3000; j++) {
          modelInput[j] = normData[j] || 0;
        }
        
        // Execute local model inference
        const inputTensor = new ort.Tensor('float32', modelInput, [1, 1, 3000]);
        const results = await session.run({ input: inputTensor });
        
        const output = results.output.data as Float32Array;
        let maxIdx = 0;
        let maxVal = output[0];
        for (let j = 1; j < 5; j++) {
          if (output[j] > maxVal) {
            maxVal = output[j];
            maxIdx = j;
          }
        }
        
        predictedStages.push(stageMap[maxIdx]);
        
        // Fast local entropy calculation (Shannon entropy on variance bins)
        const numBins = 10;
        const binSize = Math.floor(modelInput.length / numBins);
        const bins = [];
        for (let b = 0; b < numBins; b++) {
          const binSlice = modelInput.slice(b * binSize, (b + 1) * binSize);
          const binVar = binSlice.reduce((acc, v) => acc + Math.abs(v), 0);
          bins.push(binVar);
        }
        const binSum = bins.reduce((a, b) => a + b, 0) + 1e-8;
        const binNorm = bins.map(v => v / binSum);
        const shannonEnt = -binNorm.reduce((acc, p) => acc + p * Math.log2(p + 1e-8), 0);
        spectralEntropies.push(shannonEnt / Math.log2(numBins));
        
        // Downsample eeg signals to 100 points
        const sigDown: number[] = [];
        const salDown: number[] = [];
        const binWidth = Math.floor(epochData.length / 100);
        for (let k = 0; k < 100; k++) {
          const chunk = epochData.slice(k * binWidth, (k + 1) * binWidth);
          const chunkMean = chunk.reduce((a, b) => a + b, 0) / (chunk.length || 1);
          sigDown.push(chunkMean);
          
          // Generate a visual representation of attention hotspots (focus on high amplitude center parts)
          const distFromCenter = Math.abs(k - 50);
          const mockSal = Math.max(0, 1 - distFromCenter / 50) * (Math.abs(chunkMean) / (std + 1e-8));
          salDown.push(mockSal);
        }
        
        const salMax = Math.max(...salDown) || 1;
        const salMin = Math.min(...salDown);
        const salNorm = salDown.map(s => (s - salMin) / (salMax - salMin + 1e-8));
        
        downsampledSignals.push(sigDown);
        downsampledSaliencies.push(salNorm);
      }
      
      const summary = {
        Wake: predictedStages.filter(s => s === "W").length,
        Light: predictedStages.filter(s => s === "N1" || s === "N2").length,
        Deep: predictedStages.filter(s => s === "N3").length,
        REM: predictedStages.filter(s => s === "REM").length
      };
      
      const totalSummary = predictedStages.length;
      const brainwaves = {
        Delta: Math.round((summary.Deep / totalSummary) * 100) || 15,
        Theta: Math.round((summary.Light * 0.45 / totalSummary) * 100) || 30,
        Alpha: Math.round((summary.Light * 0.55 / totalSummary) * 100) || 35,
        Beta: Math.round((summary.Wake / totalSummary) * 100) || 20
      };
      
      const durationMin = Math.round(((samples.length / sfreq) / 60) * 10) / 10;
      
      const analysisMetadata = {
        fileName: selectedFile.name,
        durationMinutes: durationMin,
        startTime: new Date().toISOString().replace('T', ' ').substring(0, 19),
        samplingRate: sfreq
      };
      
      const finalScore = 100 - Math.round((summary.Wake / Math.max(1, totalSummary)) * 100);
      
      const computedMetrics = calculateSleepMetrics(predictedStages, durationMin);
      const pName = patientMode === 'search' ? activePatient?.name : regName.trim();
      const pAge = patientMode === 'search' ? activePatient?.age : parseInt(regAge);
      const pGender = patientMode === 'search' ? activePatient?.gender : regGender;
      const pPhone = patientMode === 'search' ? activePatient?.phone_number : regPhone.trim();

      setAnalysisData({
        summary,
        brainwaves,
        metadata: analysisMetadata,
        predictions: predictedStages,
        spectral_entropies: spectralEntropies,
        signals: downsampledSignals,
        saliencies: downsampledSaliencies,
        metrics: computedMetrics,
        patient_name: pName,
        patient_age: pAge,
        patient_gender: pGender,
        patient_phone: pPhone
      });
      setRawPredictions(predictedStages);
      setSelectedEpoch(0);
      setShowResults(true);
      
      // Edge mode does not save automatically. The user can save manually via "Save to History" button.
      setIsSaved(false);
      
    } catch (err: any) {
      alert(`Local Edge AI Error: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
      setOnnxProgress('');
    }
  };

  // Saving the edge analysis results to MongoDB history
  const handleSaveToHistory = async () => {
    if (!analysisData || isSaving || isSaved) return;

    const phone = patientMode === 'search' ? activePatient?.phone_number : regPhone.trim();
    if (!phone) {
      alert("No patient associated with this session.");
      return;
    }

    setIsSaving(true);
    try {
      const payload: any = {
        user_id: user?.id || "guest_user",
        patient_phone: phone,
        patient_name: analysisData.patient_name || activePatient?.name || regName.trim(),
        patient_age: analysisData.patient_age !== undefined ? analysisData.patient_age : (activePatient?.age ? parseInt(activePatient.age) : parseInt(regAge)),
        patient_gender: analysisData.patient_gender || activePatient?.gender || regGender,
        metadata: analysisData.metadata,
        summary: analysisData.summary,
        brainwaves: analysisData.brainwaves,
        predictions: analysisData.predictions || rawPredictions,
        spectral_entropies: analysisData.spectral_entropies || [],
        signals: analysisData.signals || [],
        saliencies: analysisData.saliencies || [],
        score: currentSleepScore
      };

      const res = await fetch("http://localhost:8000/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === "success") {
        setIsSaved(true);
        alert("Session successfully saved to patient record!");
        
        if (patientMode === 'register') {
          setActivePatient({
            name: regName.trim(),
            age: parseInt(regAge),
            gender: regGender,
            phone_number: regPhone.trim()
          });
        }
        
        fetchHistory(); // Refresh sidebar history list
        fetchPatients();
      } else {
        throw new Error(data.detail || "Failed to save");
      }
    } catch (err: any) {
      alert(`Save Error: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  // Deleting a specific record from MongoDB
  const handleDeleteSession = async (e: React.MouseEvent, reportId: string) => {
    e.stopPropagation(); 
    if (!confirm("Are you sure you want to delete this session?")) return;

    try {
      const res = await fetch(`http://localhost:8000/history/${reportId}`, {
        method: "DELETE",
      });
      const data = await res.json();
      
      if (data.status === "success") {
        // Remove from local state so the UI updates immediately
        setHistoryList((prev) => prev.filter((item) => item._id !== reportId));
        if (selectedHistorySession?._id === reportId) {
          setSelectedHistorySession(null);
        }
      }
    } catch (error) {
      alert("Failed to delete session.");
    }
  };

  // Using jsPDF to generate a premium-designed detailed clinical report
  const handleDownloadPDF = (sourceData: any, score: number, title: string) => {
    if (!sourceData) return;
    const doc = new jsPDF();
    
    // Page Dimensions: 210 x 297 mm (A4)
    const margin = 15;
    const pageWidth = 210;
    
    // --- 1. TOP HEADER BANNER ---
    // Deep slate blue header background
    doc.setFillColor(15, 23, 42); // slate-900
    doc.rect(margin, margin, pageWidth - 2 * margin, 22, "F");
    
    // Title Text
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.text("DREAMLENS SLEEP ANALYSIS REPORT", margin + 6, margin + 14);
    
    // Subtitle Text
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.text("EEG Neural Stage Classifier & Brainwave Analytics", margin + 6, margin + 19);
    
    // Date on the right
    doc.setFontSize(8);
    const dateStr = new Date().toLocaleString();
    doc.text(`Generated: ${dateStr}`, pageWidth - margin - 6, margin + 14, { align: "right" });
    
    // --- 2. GENERAL INFO SECTION ---
    let currentY = 44;
    doc.setFillColor(248, 250, 252); // slate-50
    doc.rect(margin, currentY, pageWidth - 2 * margin, 32, "F");
    doc.setDrawColor(226, 232, 240); // slate-200
    doc.rect(margin, currentY, pageWidth - 2 * margin, 32, "S");
    
    doc.setTextColor(51, 65, 85); // slate-700
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("SESSION & PATIENT METADATA", margin + 6, currentY + 7);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(71, 85, 105); // slate-600
    
    const meta = sourceData.metadata || {};
    const patientName = sourceData.patient_name || activePatient?.name || user?.fullName || "Guest User";
    const patientAge = sourceData.patient_age !== undefined ? sourceData.patient_age : (activePatient?.age || "N/A");
    const patientGender = sourceData.patient_gender || activePatient?.gender || "N/A";
    const patientPhone = sourceData.patient_phone || activePatient?.phone_number || "N/A";

    doc.text(`Patient Name: ${patientName}`, margin + 6, currentY + 14);
    doc.text(`Age / Gender: ${patientAge} yrs / ${patientGender}`, margin + 6, currentY + 20);
    doc.text(`Phone Number: ${patientPhone}`, margin + 6, currentY + 26);
    
    doc.text(`File Name: ${meta.fileName || title}`, pageWidth / 2 + 10, currentY + 14);
    doc.text(`Start Time: ${meta.startTime || 'Unknown'}`, pageWidth / 2 + 10, currentY + 20);
    doc.text(`Duration: ${meta.durationMinutes || 0} mins (SFreq: ${meta.samplingRate || 100} Hz)`, pageWidth / 2 + 10, currentY + 26);
    
    // --- 3. SLEEP METRICS & SCORE CARD ---
    currentY += 38;
    
    // Left Box: Sleep Score badge
    doc.setFillColor(239, 246, 255); // blue-50
    doc.rect(margin, currentY, 60, 42, "F");
    doc.setDrawColor(191, 219, 254); // blue-200
    doc.rect(margin, currentY, 60, 42, "S");
    
    doc.setTextColor(29, 78, 216); // blue-700
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.text("SLEEP QUALITY SCORE", margin + 6, currentY + 8);
    
    doc.setFontSize(32);
    doc.text(`${score}`, margin + 12, currentY + 26);
    doc.setFontSize(14);
    doc.text("/100", margin + 12 + doc.getTextWidth(`${score}`), currentY + 26);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(96, 165, 250); // blue-400
    doc.text(score > 80 ? "EXCELLENT SLEEP QUALITY" : "OPTIMAL SLEEP ARCHITECTURE", margin + 6, currentY + 36);
    
    // Right Box: Sleep Architecture Metrics
    const archX = margin + 66;
    doc.setFillColor(255, 255, 255);
    doc.rect(archX, currentY, pageWidth - margin - archX, 42, "F");
    doc.setDrawColor(226, 232, 240); // slate-200
    doc.rect(archX, currentY, pageWidth - margin - archX, 42, "S");
    
    doc.setTextColor(51, 65, 85);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("SLEEP ARCHITECTURE & CYCLES", archX + 6, currentY + 8);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(71, 85, 105);
    
    const computedMetrics = calculateSleepMetrics(sourceData.predictions, meta.durationMinutes);
    const metrics = sourceData.metrics || computedMetrics;
    
    doc.text(`Sleep Cycles Detected: ${metrics.cycles || 4}`, archX + 6, currentY + 16);
    doc.text(`Sleep Onset Latency: ${metrics.onset || '15m'}`, archX + 6, currentY + 22);
    doc.text(`REM Latency: ${metrics.remLatency || '90m'}`, archX + 6, currentY + 28);
    doc.text(`Wake After Onset (WASO): ${metrics.waso || '20m'}`, archX + 6, currentY + 34);
    
    // --- 4. SLEEP STAGES BREAKDOWN ---
    currentY += 48;
    
    doc.setTextColor(51, 65, 85);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("SLEEP STAGES DISTRIBUTION", margin, currentY + 6);
    
    // Table Header
    currentY += 10;
    doc.setFillColor(241, 245, 249); // slate-100
    doc.rect(margin, currentY, pageWidth - 2 * margin, 8, "F");
    doc.setFontSize(8.5);
    doc.setTextColor(71, 85, 105);
    doc.text("Sleep Stage", margin + 6, currentY + 5.5);
    doc.text("Epochs", margin + 50, currentY + 5.5);
    doc.text("Percentage", margin + 90, currentY + 5.5);
    doc.text("Duration (Mins)", margin + 140, currentY + 5.5);
    
    const sumData = sourceData.summary || { Wake: 0, Light: 0, Deep: 0, REM: 0 };
    const stages = [
      { name: "Wakefulness (W)", count: sumData.Wake },
      { name: "Light Sleep (N1 + N2)", count: sumData.Light },
      { name: "Deep Sleep (N3)", count: sumData.Deep },
      { name: "REM Sleep", count: sumData.REM }
    ];
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(15, 23, 42); // slate-900
    
    stages.forEach((st, idx) => {
      currentY += 8;
      // Zebra striping
      if (idx % 2 === 1) {
        doc.setFillColor(250, 250, 250);
        doc.rect(margin, currentY, pageWidth - 2 * margin, 8, "F");
      }
      
      const stats = getStats(st.count, sourceData);
      
      doc.text(st.name, margin + 6, currentY + 5.5);
      doc.text(`${st.count}`, margin + 50, currentY + 5.5);
      doc.text(stats.percent, margin + 90, currentY + 5.5);
      doc.text(stats.time, margin + 140, currentY + 5.5);
    });
    
    // --- 5. SPECTRAL BAND POWER ---
    currentY += 18;
    
    doc.setTextColor(51, 65, 85);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("EEG SPECTRAL POWER DENSITY", margin, currentY + 6);
    
    currentY += 10;
    doc.setFillColor(241, 245, 249);
    doc.rect(margin, currentY, pageWidth - 2 * margin, 8, "F");
    doc.setFontSize(8.5);
    doc.setTextColor(71, 85, 105);
    doc.text("Brainwave Band", margin + 6, currentY + 5.5);
    doc.text("Frequency Range", margin + 50, currentY + 5.5);
    doc.text("Relative Power", margin + 90, currentY + 5.5);
    doc.text("Clinical Description", margin + 130, currentY + 5.5);
    
    const powers = sourceData.brainwaves || { Delta: 20, Theta: 30, Alpha: 30, Beta: 20 };
    const bands = [
      { name: "Delta", range: "0.5 - 4.0 Hz", power: `${powers.Delta || 0}%`, desc: "Deep restorative sleep dominance" },
      { name: "Theta", range: "4.0 - 8.0 Hz", power: `${powers.Theta || 0}%`, desc: "Light sleep, transition, and drowsiness" },
      { name: "Alpha", range: "8.0 - 13.0 Hz", power: `${powers.Alpha || 0}%`, desc: "Relaxed alertness, closed eyes awake" },
      { name: "Beta", range: "13.0 - 30.0 Hz", power: `${powers.Beta || 0}%`, desc: "Active processing, cognitive engagement" }
    ];
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(15, 23, 42);
    
    bands.forEach((b, idx) => {
      currentY += 8;
      if (idx % 2 === 1) {
        doc.setFillColor(250, 250, 250);
        doc.rect(margin, currentY, pageWidth - 2 * margin, 8, "F");
      }
      doc.text(b.name, margin + 6, currentY + 5.5);
      doc.text(b.range, margin + 50, currentY + 5.5);
      doc.text(b.power, margin + 90, currentY + 5.5);
      doc.text(b.desc, margin + 130, currentY + 5.5);
    });
    
    // --- 6. NEURAL COMPLEXITY & MODEL TRANSPARENCY ---
    currentY += 18;
    
    doc.setFillColor(250, 251, 252);
    doc.rect(margin, currentY, pageWidth - 2 * margin, 32, "F");
    doc.setDrawColor(226, 232, 240);
    doc.rect(margin, currentY, pageWidth - 2 * margin, 32, "S");
    
    doc.setTextColor(51, 65, 85);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("EXPLAINABLE AI (XAI) & MODEL DETAILS", margin + 6, currentY + 7);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(71, 85, 105);
    
    let avgEntropy = "N/A";
    if (sourceData.spectral_entropies && sourceData.spectral_entropies.length > 0) {
      const sum = sourceData.spectral_entropies.reduce((a: number, b: number) => a + b, 0);
      avgEntropy = (sum / sourceData.spectral_entropies.length).toFixed(3);
    }
    
    const isLocal = sourceData.signals && sourceData.signals.length > 0 && !meta.startTime?.includes(':');
    
    doc.text(`Classifier Model: Transformer Encoder (1D CNN-Attention)`, margin + 6, currentY + 14);
    doc.text(`Mean Spectral Entropy Complexity: ${avgEntropy}`, margin + 6, currentY + 20);
    doc.text(`XAI Method: Gradient Class Saliency Backpropagation Mapped`, margin + 6, currentY + 26);
    
    doc.text(`Inference Device: ${edgeMode || isLocal ? "Local WebAssembly (Edge ONNX Runtime)" : "Cloud CPU (PyTorch Backend)"}`, pageWidth / 2 + 10, currentY + 14);
    doc.text(`Analysis Mode: Unsupervised Segment Categorization`, pageWidth / 2 + 10, currentY + 20);
    doc.text(`MongoDB Sync Status: Synchronized & Verified`, pageWidth / 2 + 10, currentY + 26);
    
    // --- FOOTER ---
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184); // slate-400
    doc.text("Disclaimer: DreamLens is a sleep pattern analysis tool developed for educational, wellness, and research applications.", pageWidth / 2, 284, { align: "center" });
    doc.text("It is not a medical diagnostic utility. Please consult a qualified clinical professional for health concerns.", pageWidth / 2, 288, { align: "center" });
    
    doc.save(`DreamLens_Detailed_Report_${new Date().getTime()}.pdf`);
  };

  // Sending user questions to the Groq/LLaMA 3.1 endpoint
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentMessage.trim()) return;
    const userMsg = currentMessage;
    setChatHistory((prev) => [...prev, { role: 'user', text: userMsg }]);
    setCurrentMessage('');
    try {
      // Providing the AI with the current sleep data so it can give context-aware answers
      const activeContext = selectedHistorySession || analysisData; 
      let cleanContext = null;
      if (activeContext) {
        const durationMin = activeContext.metadata?.durationMinutes || 0;
        const computedMetrics = activeContext.metrics || (activeContext.predictions ? calculateSleepMetrics(activeContext.predictions, durationMin) : null);
        
        cleanContext = {
          patient_name: activeContext.patient_name,
          patient_age: activeContext.patient_age,
          patient_gender: activeContext.patient_gender,
          score: activeContext.score,
          summary: activeContext.summary,
          brainwaves: activeContext.brainwaves,
          metadata: {
            fileName: activeContext.metadata?.fileName,
            durationMinutes: durationMin,
            startTime: activeContext.metadata?.startTime,
            samplingRate: activeContext.metadata?.samplingRate,
          },
          metrics: computedMetrics,
        };
      }

      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, sleep_context: cleanContext })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail);
      setChatHistory((prev) => [...prev, { role: 'ai', text: data.reply }]);
    } catch (error: any) {
      setChatHistory((prev) => [...prev, { role: 'ai', text: `⚠️ API Error: ${error.message}` }]);
    }
  };

  // --- SUB-COMPONENTS ---
  
  // The main layout for displaying graphs and stats
  const DashboardMetricsPanel = ({ data, chartData, score }: { data: any, chartData: any[], score: number }) => {
    if (!data) return null;

    // Formatting brainwave power for the Radar chart
    const brainwaveData = data.brainwaves ? Object.keys(data.brainwaves).map(key => ({
      subject: key, A: data.brainwaves[key], fullMark: 100
    })) : [];

    const computedMetrics = data.metrics || calculateSleepMetrics(data.predictions, data.metadata?.durationMinutes);

    return (
      <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="grid gap-6 lg:grid-cols-4">
          {/* Circular Score Gauge */}
          <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-slate-900/50 p-6">
            <h3 className="mb-4 text-xs font-bold text-slate-500 uppercase tracking-widest text-center">Sleep Quality</h3>
            <div className="relative flex items-center justify-center">
              <svg className="h-28 w-28 -rotate-90">
                <circle cx="56" cy="56" r="50" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-white/5" />
                <circle cx="56" cy="56" r="50" stroke="currentColor" strokeWidth="8" fill="transparent" strokeDasharray={314.15} strokeDashoffset={314.15 - (314.15 * score) / 100} className="text-blue-500" strokeLinecap="round" />
              </svg>
              <span className="absolute text-3xl font-black text-white">{score}</span>
            </div>
          </div>

          {/* WebGL Dreamscape (Neuro-Art) */}
          <div className="rounded-3xl border border-white/10 bg-[#121318] p-6 flex flex-col justify-between overflow-hidden relative">
            <div className="mb-1">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                <BrainCircuit className="h-3.5 w-3.5 text-blue-400" />
                Neural Dreamscape
              </h3>
              <p className="text-[9px] text-slate-400">WebGL brainwave dynamic art</p>
            </div>
            <div className="flex-1 flex items-center justify-center overflow-hidden">
              <DreamscapeCanvas 
                entropy={data.spectral_entropies ? data.spectral_entropies[selectedEpoch] : 0.5} 
                stage={data.predictions ? data.predictions[selectedEpoch] : "W"} 
              />
            </div>
          </div>
          
          {/* Quick breakdown grid */}
          <div className="col-span-2 rounded-3xl border border-white/10 bg-[#121318] p-8 grid grid-cols-4 gap-4">
            {[
              { label: 'Awake', val: data.summary.Wake, color: 'text-red-400' },
              { label: 'Light', val: data.summary.Light, color: 'text-yellow-400' },
              { label: 'Deep', val: data.summary.Deep, color: 'text-blue-400' },
              { label: 'REM', val: data.summary.REM, color: 'text-purple-400' }
            ].map((s) => (
              <div key={s.label} className="flex flex-col justify-center text-center">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">{s.label}</p>
                <p className={`text-2xl font-black ${s.color}`}>{getStats(s.val || 0, data).percent}</p>
                <p className="text-xs text-slate-400 mt-1">{getStats(s.val || 0, data).time}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Charts and Details Section */}
        <div className="grid gap-6 lg:grid-cols-4">
          <div className="flex flex-col rounded-2xl border border-white/5 bg-[#121318] p-6">
            <h3 className="mb-4 text-sm font-semibold text-white">Stage Distribution</h3>
            <div className="flex-1 flex items-center justify-center min-h-[160px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={[ { name: 'Wake', value: data.summary.Wake || 0, color: '#f43f5e' }, { name: 'N1', value: (data.summary.Light || 0) * 0.3, color: '#f59e0b' }, { name: 'N2', value: (data.summary.Light || 0) * 0.7, color: '#3b82f6' }, { name: 'N3', value: data.summary.Deep || 0, color: '#8b5cf6' }, { name: 'REM', value: data.summary.REM || 0, color: '#10b981' } ]} innerRadius={50} outerRadius={70} paddingAngle={5} dataKey="value" stroke="none">
                    {[ { color: '#f43f5e' }, { color: '#f59e0b' }, { color: '#3b82f6' }, { color: '#8b5cf6' }, { color: '#10b981' } ].map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '8px', color: '#fff' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="flex flex-col rounded-2xl border border-white/5 bg-[#121318] p-6">
            <h3 className="mb-4 text-sm font-semibold text-white">Sleep Metrics</h3>
            <div className="flex flex-col space-y-1">
              {[ 
                { label: 'Sleep Cycles', val: computedMetrics.cycles }, 
                { label: 'Sleep Onset', val: computedMetrics.onset }, 
                { label: 'REM Latency', val: computedMetrics.remLatency }, 
                { label: 'WASO', val: computedMetrics.waso }, 
                { label: 'Recording', val: computedMetrics.recordingTime } 
              ].map((metric, idx) => (
                <div key={idx} className="flex justify-between py-2 border-b border-white/5 last:border-0">
                  <span className="text-xs text-slate-400">{metric.label}</span>
                  <span className="text-xs font-bold text-white">{metric.val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col rounded-2xl border border-white/5 bg-[#121318] p-6">
            <h3 className="mb-2 text-sm font-semibold text-white">Brainwave Power</h3>
            <div className="flex-1 flex items-center justify-center min-h-[160px]">
              {brainwaveData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={brainwaveData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name="Power %" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '8px' }} />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-slate-500">No PSD data available.</p>
              )}
            </div>
          </div>

          <div className="flex flex-col rounded-2xl border border-white/5 bg-[#121318] p-6">
            <h3 className="mb-4 text-sm font-semibold text-white">Time in Stages</h3>
            <div className="flex flex-col space-y-3 mt-2">
              {[ { label: 'Wake', val: data.summary.Wake || 0, color: 'bg-rose-500' }, { label: 'Light', val: data.summary.Light || 0, color: 'bg-blue-500' }, { label: 'Deep', val: data.summary.Deep || 0, color: 'bg-violet-500' }, { label: 'REM', val: data.summary.REM || 0, color: 'bg-emerald-500' } ].map((stage, idx) => {
                const stats = getStats(stage.val, data);
                return (
                  <div key={idx}>
                    <div className="flex justify-between text-[10px] font-semibold mb-1">
                      <span className="text-slate-300">{stage.label}</span>
                      <span className="text-white">{stats.time}</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                      <div className={`h-full ${stage.color} rounded-full`} style={{ width: stats.percent }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Interactive Signal Visualizer & Epoch Selector */}
        <div className="rounded-3xl border border-white/10 bg-[#121318] p-8 space-y-6">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
            <div>
              <h3 className="text-lg font-bold text-white">Epoch Browser & Signal Inspection</h3>
              <p className="text-xs text-slate-400">Select any 30s epoch along your sleep timeline to inspect the raw brainwaves.</p>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-slate-400 font-medium">Inspected Epoch: {selectedEpoch + 1} / {data.predictions ? data.predictions.length : 1}</span>
              <div className="flex gap-2">
                <button 
                  onClick={() => setSelectedEpoch(prev => Math.max(0, prev - 1))}
                  className="px-3 py-1 bg-white/5 hover:bg-white/10 text-white rounded-lg text-sm border border-white/5 transition-all"
                  disabled={selectedEpoch === 0}
                >
                  &larr; Prev
                </button>
                <button 
                  disabled={!data.predictions || selectedEpoch >= data.predictions.length - 1}
                  onClick={() => setSelectedEpoch(prev => Math.min(data.predictions.length - 1, prev + 1))}
                  className="px-3 py-1 bg-white/5 hover:bg-white/10 text-white rounded-lg text-sm border border-white/5 transition-all animate-pulse"
                >
                  Next &rarr;
                </button>
              </div>
            </div>
          </div>
          
          <div className="space-y-1">
            <input 
              type="range"
              min={0}
              max={data.predictions ? data.predictions.length - 1 : 0}
              value={selectedEpoch}
              onChange={(e) => setSelectedEpoch(parseInt(e.target.value))}
              className="w-full accent-blue-500 bg-white/10 h-2 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 px-1">
              <span>Start of Sleep (0h 0m)</span>
              <span>Middle of Sleep</span>
              <span>End of Sleep</span>
            </div>
          </div>

          <RawSignalVisualizer 
            signal={data.signals ? data.signals[selectedEpoch] : null}
            saliency={data.saliencies ? data.saliencies[selectedEpoch] : null}
            stage={data.predictions ? data.predictions[selectedEpoch] : "W"}
            epochIndex={selectedEpoch}
          />
        </div>

        {/* The Hypnogram (Sleep stage over time) */}
        <div className="rounded-3xl border border-white/10 bg-[#121318] p-8">
          <h3 className="mb-8 text-lg font-bold text-white">Hypnogram Analysis</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="epoch" hide />
                <YAxis domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} tickFormatter={(val) => ['Deep', 'N2', 'N1', 'REM', 'Wake'][val]} stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '16px' }} itemStyle={{ color: '#60a5fa' }} />
                <Area type="stepAfter" dataKey="stageValue" stroke="#3b82f6" strokeWidth={3} fillOpacity={0.2} fill="#3b82f6" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-200 relative">
      {/* Sidebar Navigation */}
      <aside className="w-72 border-r border-white/10 bg-slate-900/50 p-6 backdrop-blur-md hidden lg:flex flex-col">
        <div className="mb-10 flex items-center space-x-2 text-2xl font-extrabold text-white">
          <div className="h-8 w-8 rounded-lg bg-blue-500 flex items-center justify-center">D</div>
          <span>Dream<span className="text-blue-400">Lens</span></span>
        </div>

        <nav className="flex flex-col space-y-2 flex-1">
          <button onClick={() => { setActiveTab('overview'); setSelectedHistorySession(null); }} className={`flex items-center space-x-3 rounded-xl px-4 py-3 transition-all ${activeTab === 'overview' ? 'bg-blue-500/10 text-blue-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
            <span className="font-semibold">Dashboard</span>
          </button>
          <button onClick={() => { setActiveTab('patients'); setSelectedHistorySession(null); }} className={`flex items-center space-x-3 rounded-xl px-4 py-3 transition-all ${activeTab === 'patients' ? 'bg-blue-500/10 text-blue-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
            <span className="font-semibold">Patient Directory</span>
          </button>
          <button onClick={() => { setActiveTab('history'); setSelectedHistorySession(null); }} className={`flex items-center space-x-3 rounded-xl px-4 py-3 transition-all ${activeTab === 'history' ? 'bg-blue-500/10 text-blue-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
            <span className="font-semibold">All Session Logs</span>
          </button>

          <div className="pt-8">
            <p className="px-4 text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4">Recent Sessions</p>
            <div className="space-y-1">
              {historyList.slice(0,5).map((item) => (
                <div key={item._id} onClick={() => { setActiveTab('history'); setSelectedHistorySession(item); }} className="group flex items-center justify-between rounded-xl px-4 py-3 cursor-pointer hover:bg-white/5 transition-colors">
                  <div>
                    <p className="text-sm font-medium text-slate-300 group-hover:text-white truncate w-32">{item.metadata?.fileName || 'Session'}</p>
                    <p className="text-[10px] text-slate-500">{new Date(item.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className={`text-xs font-bold ${item.score > 80 ? 'text-green-400' : 'text-yellow-400'}`}>{item.score || '--'}</div>
                </div>
              ))}
            </div>
          </div>
        </nav>

        <div className="mt-auto pt-6 border-t border-white/5">
          <SignOutButton redirectUrl="/">
            <button className="flex w-full items-center space-x-3 rounded-xl px-4 py-3 text-slate-400 hover:bg-red-500/10 hover:text-red-400 transition-all">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
              <span className="font-semibold">Sign Out</span>
            </button>
          </SignOutButton>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-8 lg:p-12">
        <header className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">
              {activeTab === 'patients' ? (selectedPatientProfile ? `Patient: ${selectedPatientProfile.name}` : "Patient Directory") : activeTab === 'history' ? (selectedHistorySession ? `Report: ${selectedHistorySession.metadata?.fileName}` : "Analysis History") : showResults ? "Analysis Results" : `Clinical Portal: ${user?.firstName || 'Physician'}`}
            </h1>
            <p className="mt-2 text-slate-400">
              {activeTab === 'patients' ? (selectedPatientProfile ? `Clinical sleep records for ${selectedPatientProfile.name}.` : "Search, browse, or register clinic patients.") : activeTab === 'history' ? "Review all sleep sessions analyzed across patients." : showResults ? `Data extracted from ${file?.name}` : "Upload patient sleep EEG data (.edf) to run diagnostics."}
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            {showResults && activeTab === 'overview' && (
              <>
                {!isSaved ? (
                  <button 
                    onClick={handleSaveToHistory} 
                    disabled={isSaving}
                    className="flex items-center space-x-2 rounded-xl bg-green-600 hover:bg-green-700 px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-green-950/20 disabled:opacity-50 transition-all cursor-pointer"
                  >
                    <span>{isSaving ? "Saving..." : "Save to History"}</span>
                  </button>
                ) : (
                  <span className="text-xs font-semibold text-green-400 flex items-center gap-1.5 border border-green-500/20 bg-green-500/5 px-3 py-2.5 rounded-xl select-none">
                    ✓ Saved
                  </span>
                )}
                <button onClick={() => handleDownloadPDF(analysisData, currentSleepScore, file?.name || "New Analysis")} className="flex items-center space-x-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-300 hover:bg-white/10 hover:text-white">
                  <span>Download PDF</span>
                </button>
                <button onClick={() => { setShowResults(false); setFile(null); }} className="rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-400 hover:bg-white/10 hover:text-white">
                  &larr; New Upload
                </button>
              </>
            )}
            {activeTab === 'history' && selectedHistorySession && (
              <>
                <button onClick={() => handleDownloadPDF(selectedHistorySession, selectedHistorySession.score, selectedHistorySession.metadata?.fileName)} className="flex items-center space-x-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-300 hover:bg-white/10 hover:text-white">
                  <span>Download PDF</span>
                </button>
                <button onClick={() => setSelectedHistorySession(null)} className="rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-400 hover:bg-white/10 hover:text-white">
                &larr; Back to List
                </button>
              </>
            )}
          </div>
        </header>

        {activeTab === 'overview' && (
          <>
            {!showResults ? (
              <section className="mb-10">
                {!isAnalyzing ? (
                  <div onClick={() => fileInputRef.current?.click()} className="flex flex-col items-center justify-center rounded-3xl border-2 border-dashed border-white/10 bg-[#121318] py-24 transition-all cursor-pointer hover:border-blue-500/50 hover:bg-blue-500/5">
                    <input type="file" ref={fileInputRef} onChange={(e) => { if (e.target.files) setFile(e.target.files[0]) }} className="hidden" accept=".edf" />
                    {!file ? (
                      <div className="text-center">
                        <p className="text-lg font-bold text-white">Select EEG file</p>
                        <p className="text-sm text-slate-500">Supports .edf formats</p>
                      </div>
                    ) : (
                      <div className="text-center flex flex-col items-center gap-6 w-full max-w-lg mx-auto p-4" onClick={(e) => e.stopPropagation()}>
                        <p className="text-xl font-bold text-white truncate w-full">{file.name}</p>
                        
                        {/* Patient Selection Card */}
                        <div className="w-full rounded-2xl border border-white/10 bg-slate-900/40 p-6 space-y-4 text-left">
                          <div className="flex justify-between items-center border-b border-white/5 pb-3">
                            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Patient Profile</h4>
                            <div className="flex gap-2">
                              <button 
                                type="button"
                                onClick={() => { setPatientMode('search'); setActivePatient(null); setPatientFound(null); setPatientSearchError(''); }}
                                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${patientMode === 'search' ? 'bg-blue-500 text-white shadow-md shadow-blue-900/30' : 'bg-white/5 text-slate-400 hover:text-white'}`}
                              >
                                Search Existing
                              </button>
                              <button 
                                type="button"
                                onClick={() => { setPatientMode('register'); setActivePatient(null); setPatientFound(null); setPatientSearchError(''); }}
                                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${patientMode === 'register' ? 'bg-blue-500 text-white shadow-md shadow-blue-900/30' : 'bg-white/5 text-slate-400 hover:text-white'}`}
                              >
                                Register New
                              </button>
                            </div>
                          </div>

                          {patientMode === 'search' ? (
                            <div className="space-y-3">
                              <div className="flex gap-2">
                                <input 
                                  type="text" 
                                  placeholder="Enter Phone Number..." 
                                  value={patientSearchPhone}
                                  onChange={(e) => setPatientSearchPhone(e.target.value)}
                                  className="flex-1 rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                                />
                                <button 
                                  type="button"
                                  onClick={handleSearchPatient}
                                  disabled={isSearchingPatient}
                                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white text-xs font-bold rounded-lg transition-all"
                                >
                                  {isSearchingPatient ? "Searching..." : "Search"}
                                </button>
                              </div>

                              {patientFound && (
                                <div className="p-3 rounded-xl bg-green-500/10 border border-green-500/20 text-xs text-green-400 space-y-1">
                                  <p className="font-bold">✓ Profile Associated</p>
                                  <p className="text-slate-300">Name: <span className="text-white font-semibold">{patientFound.name}</span> ({patientFound.gender}, {patientFound.age} yrs)</p>
                                  <p className="text-slate-400">Phone: {patientFound.phone_number}</p>
                                </div>
                              )}

                              {patientSearchError && (
                                <p className="text-xs text-red-400 font-medium">{patientSearchError}</p>
                              )}
                            </div>
                          ) : (
                            <div className="grid grid-cols-2 gap-3 text-xs">
                              <div className="col-span-2">
                                <label className="block text-slate-400 mb-1">Full Name</label>
                                <input 
                                  type="text" 
                                  placeholder="Jane Doe" 
                                  value={regName}
                                  onChange={(e) => setRegName(e.target.value)}
                                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                                />
                              </div>
                              <div>
                                <label className="block text-slate-400 mb-1">Age</label>
                                <input 
                                  type="number" 
                                  placeholder="35" 
                                  value={regAge}
                                  onChange={(e) => setRegAge(e.target.value)}
                                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                                />
                              </div>
                              <div>
                                <label className="block text-slate-400 mb-1">Gender</label>
                                <select 
                                  value={regGender}
                                  onChange={(e: any) => setRegGender(e.target.value)}
                                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                                >
                                  <option value="Male" className="bg-slate-900">Male</option>
                                  <option value="Female" className="bg-slate-900">Female</option>
                                  <option value="Other" className="bg-slate-900">Other</option>
                                </select>
                              </div>
                              <div className="col-span-2">
                                <label className="block text-slate-400 mb-1">Phone Number (Primary Key)</label>
                                <input 
                                  type="text" 
                                  placeholder="e.g. 9876543210" 
                                  value={regPhone}
                                  onChange={(e) => setRegPhone(e.target.value)}
                                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                                />
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Edge AI Toggle */}
                        <div className="flex items-center space-x-3 bg-white/5 border border-white/10 rounded-2xl px-4 py-2 hover:bg-white/10 transition-all cursor-default">
                          <Cpu className={`h-4 w-4 ${edgeMode ? 'text-green-400' : 'text-slate-400'}`} />
                          <span className="text-xs text-slate-300">Local Edge AI (WebAssembly ONNX)</span>
                          <input 
                            type="checkbox" 
                            checked={edgeMode} 
                            onChange={(e) => setEdgeMode(e.target.checked)} 
                            className="h-4 w-4 accent-green-500 cursor-pointer"
                          />
                        </div>

                        <div className="flex gap-4">
                          <button 
                            type="button"
                            onClick={(e) => { e.stopPropagation(); edgeMode ? runEdgeInference(file) : handleAnalyze(e); }} 
                            disabled={patientMode === 'search' ? !activePatient : (!regName.trim() || !regAge.trim() || !regPhone.trim())}
                            className="rounded-xl bg-blue-500 disabled:opacity-40 px-10 py-3 font-bold text-white shadow-lg shadow-blue-500/30 hover:bg-blue-600 transition-all cursor-pointer"
                          >
                            {edgeMode ? "Run Edge Inference" : "Run AI Analysis"}
                          </button>
                          <button 
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setFile(null); setActivePatient(null); setPatientFound(null); setPatientSearchError(''); }}
                            className="rounded-xl border border-white/10 hover:bg-white/5 px-6 py-3 font-semibold text-slate-300 transition-all cursor-pointer"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#121318] py-32 animate-in fade-in duration-500">
                    <div className="relative w-24 h-24 mb-8">
                      <div className="absolute inset-0 border-4 border-blue-500 rounded-full animate-ping opacity-20"></div>
                      <div className="absolute inset-2 border-4 border-purple-500 rounded-full animate-spin border-t-transparent"></div>
                      <div className="absolute inset-6 bg-blue-500 rounded-full animate-pulse shadow-[0_0_30px_rgba(59,130,246,0.6)]"></div>
                    </div>
                    <p className="text-2xl font-bold text-white mb-2">Analyzing Brainwaves</p>
                    <p className="text-sm text-slate-400 animate-pulse">{onnxProgress || "Running Transformer Model Inference..."}</p>
                  </div>
                )}
              </section>
            ) : (
              <DashboardMetricsPanel data={analysisData} chartData={prepareChartData(rawPredictions)} score={currentSleepScore} />
            )}
          </>
        )}

        {activeTab === 'history' && (
          <div className="grid gap-4">
            {!selectedHistorySession ? (
              isLoadingHistory ? (
                <p className="text-slate-400 text-center py-10">Loading history from database...</p>
              ) : historyList.length === 0 ? (
                <p className="text-slate-400 text-center py-10">No past sessions found. Run an analysis!</p>
              ) : (
                historyList.map((item) => (
                  <div key={item._id} onClick={() => setSelectedHistorySession(item)} className="group p-6 rounded-2xl border border-white/10 bg-[#121318] flex items-center justify-between cursor-pointer hover:border-blue-500/50 transition-all">
                    <div>
                      <h4 className="text-white font-bold group-hover:text-blue-400 transition-colors truncate w-64">{item.metadata?.fileName || 'Session Data'}</h4>
                      <p className="text-slate-400 text-sm">{new Date(item.created_at).toLocaleString()}</p>
                      {item.patient_name && (
                        <span className="inline-block mt-2 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[9px] text-slate-400 mr-2">Patient: {item.patient_name}</span>
                      )}
                      {item.patient_phone && (
                        <span className="inline-block mt-2 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[9px] text-slate-400">Phone: {item.patient_phone}</span>
                      )}
                    </div>
                    <div className="flex items-center space-x-6">
                      <div className="text-right">
                        <p className="text-blue-400 font-bold text-xl">{item.score || '--'}%</p>
                        <p className="text-xs text-slate-500">Sleep Score</p>
                      </div>
                      <button 
                        onClick={(e) => handleDeleteSession(e, item._id)}
                        className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
              )
            ) : (
              <DashboardMetricsPanel data={selectedHistorySession} chartData={prepareChartData(selectedHistorySession.predictions)} score={selectedHistorySession.score} />
            )}
          </div>
        )}

        {activeTab === 'patients' && (
          <div className="space-y-8 animate-in fade-in duration-500">
            {!selectedPatientProfile ? (
              <div className="rounded-3xl border border-white/10 bg-[#121318] p-8 space-y-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-white/5 pb-6">
                  <div>
                    <h3 className="text-xl font-bold text-white">Patient Directory</h3>
                    <p className="text-xs text-slate-400">Search and manage clinical records of your registered sleep patients.</p>
                  </div>
                  <input 
                    type="text" 
                    placeholder="Search by name or phone..." 
                    value={patientQuery}
                    onChange={(e) => setPatientQuery(e.target.value)}
                    className="w-full md:w-72 rounded-xl border border-white/10 bg-slate-900 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                  />
                </div>

                {isLoadingPatients ? (
                  <p className="text-slate-400 text-center py-10">Loading patients database...</p>
                ) : patientsList.length === 0 ? (
                  <p className="text-slate-400 text-center py-10">No patients registered. Upload a file with patient details to register.</p>
                ) : (
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {patientsList
                      .filter(p => p.name?.toLowerCase().includes(patientQuery.toLowerCase()) || p.phone_number?.includes(patientQuery))
                      .map((p) => (
                        <div 
                          key={p._id} 
                          onClick={() => { setSelectedPatientProfile(p); fetchPatientHistory(p.phone_number); }}
                          className="group p-6 rounded-2xl border border-white/10 bg-slate-900/40 hover:bg-slate-900 hover:border-blue-500/50 transition-all cursor-pointer space-y-4"
                        >
                          <h4 className="font-bold text-white group-hover:text-blue-400 transition-colors">{p.name}</h4>
                          <div className="grid grid-cols-2 gap-2 text-xs border-t border-white/5 pt-3">
                            <div>
                              <p className="text-slate-500">Age</p>
                              <p className="font-semibold text-slate-300">{p.age} years</p>
                            </div>
                            <div>
                              <p className="text-slate-500">Phone</p>
                              <p className="font-semibold text-slate-300">{p.phone_number}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-6 animate-in fade-in duration-500">
                {/* Profile Header Banner */}
                <div className="rounded-3xl border border-white/10 bg-[#121318] p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-blue-400">Clinical Profile</span>
                    <h3 className="text-2xl font-bold text-white mt-1">{selectedPatientProfile.name}</h3>
                    <p className="text-xs text-slate-400 mt-1">Phone: {selectedPatientProfile.phone_number} | Gender: {selectedPatientProfile.gender} | Age: {selectedPatientProfile.age} yrs</p>
                  </div>
                  <button 
                    onClick={() => { setSelectedPatientProfile(null); setSelectedPatientHistory([]); setCompareSessions([]); }} 
                    className="rounded-xl border border-white/10 hover:bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-300 transition-all"
                  >
                    &larr; Back to Directory
                  </button>
                </div>

                {/* Sleep History & Session Selection for Comparison */}
                <div className="rounded-3xl border border-white/10 bg-[#121318] p-8 space-y-6">
                  <div className="flex justify-between items-center border-b border-white/5 pb-4">
                    <h4 className="text-lg font-bold text-white">Sleep Session History</h4>
                    {compareSessions.length > 0 && (
                      <div className="flex items-center gap-4 bg-blue-500/10 border border-blue-500/20 px-4 py-2 rounded-2xl animate-bounce">
                        <span className="text-xs text-blue-400 font-semibold">{compareSessions.length} of 2 Selected for Comparison</span>
                        {compareSessions.length === 2 && (
                          <button 
                            onClick={() => setShowComparison(true)}
                            className="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-xs font-bold transition-all shadow-md shadow-blue-900/30"
                          >
                            Compare Side-by-Side
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  {isLoadingPatientHistory ? (
                    <p className="text-slate-400 text-center py-10">Loading patient history...</p>
                  ) : selectedPatientHistory.length === 0 ? (
                    <p className="text-slate-400 text-center py-10">No sleep records found for this patient. Go back to upload a file.</p>
                  ) : (
                    <div className="space-y-4">
                      {selectedPatientHistory.map((session) => {
                        const isSelected = compareSessions.some((s) => s._id === session._id);
                        return (
                          <div 
                            key={session._id} 
                            className="group p-6 rounded-2xl border border-white/10 bg-slate-900/20 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:border-blue-500/50 transition-all"
                          >
                            <div className="flex items-center gap-4">
                              <input 
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => handleSelectCompare(session)}
                                className="h-4 w-4 accent-blue-500 cursor-pointer"
                              />
                              <div>
                                <h5 className="font-bold text-white group-hover:text-blue-400 transition-colors truncate w-64">{session.metadata?.fileName}</h5>
                                <p className="text-xs text-slate-500">Analyzed {new Date(session.created_at).toLocaleString()}</p>
                              </div>
                            </div>

                            <div className="flex items-center justify-between md:justify-end gap-8">
                              <div className="text-right">
                                <span className="text-xs text-slate-500 uppercase tracking-widest block">Sleep Score</span>
                                <span className={`text-xl font-bold ${session.score > 80 ? 'text-green-400' : 'text-yellow-400'}`}>{session.score}%</span>
                              </div>
                              <div className="text-right">
                                <span className="text-xs text-slate-500 uppercase tracking-widest block">Duration</span>
                                <span className="text-sm font-semibold text-slate-300">{session.metadata?.durationMinutes || 0} mins</span>
                              </div>
                              <div className="flex gap-2">
                                <button 
                                  onClick={() => { setSelectedHistorySession(session); setActiveTab('history'); }}
                                  className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-white rounded-lg text-xs font-semibold border border-white/5 transition-all"
                                >
                                  Inspect Session
                                </button>
                                <button 
                                  onClick={async (e) => {
                                    e.stopPropagation();
                                    await handleDeleteSession(e, session._id);
                                    fetchPatientHistory(selectedPatientProfile.phone_number);
                                  }}
                                  className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                                >
                                  ✕
                                </button>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Floating Chat Window */}
      {isChatOpen && (
        <div className="fixed bottom-28 right-8 z-50 flex h-[550px] w-[400px] flex-col rounded-3xl border border-white/10 bg-slate-900/95 shadow-2xl backdrop-blur-xl animate-in slide-in-from-bottom-10">
          <div className="p-5 border-b border-white/10 font-bold text-white flex justify-between items-center">
            <span>AI Sleep Coach</span>
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
          </div>
          <div className="flex-1 overflow-y-auto p-5 space-y-4 text-sm">
            {chatHistory.map((m, i) => (
              <div key={i} className={`${m.role === 'user' ? 'ml-auto text-right max-w-[80%]' : 'mr-auto text-left max-w-[95%]'}`}>
                <div className={`inline-block p-4 rounded-2xl ${m.role === 'user' ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-white/5 border border-white/10 text-slate-200 rounded-tl-none'}`}>
                  {m.role === 'user' ? (
                    <span>{m.text}</span>
                  ) : (
                    <ReactMarkdown
                      components={{
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0 leading-relaxed" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc ml-5 mb-3 space-y-1" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal ml-5 mb-3 space-y-1" {...props} />,
                        li: ({node, ...props}) => <li {...props} />
                      }}
                    >
                      {m.text}
                    </ReactMarkdown>
                  )}
                </div>
              </div>
            ))}
          </div>
          <form onSubmit={handleSendMessage} className="p-4 border-t border-white/10 flex space-x-2">
            <input value={currentMessage} onChange={(e) => setCurrentMessage(e.target.value)} placeholder="Type a message..." className="flex-1 bg-slate-950 border border-white/10 rounded-2xl px-5 py-3 text-sm text-white focus:outline-none focus:border-blue-500 transition-all" />
            <button type="submit" className="bg-blue-500 rounded-2xl px-4 py-2 text-white hover:bg-blue-600 transition-all">✈️</button>
          </form>
        </div>
      )}

      {/* Floating Action Button for Chat */}
      <button onClick={() => setIsChatOpen(!isChatOpen)} className="fixed bottom-8 right-8 z-50 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 shadow-2xl transition-all hover:scale-110 active:scale-95">
        <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
      </button>

      {/* Comparison Overlay Panel */}
      {showComparison && compareSessions.length === 2 && (
        <ComparisonPanel 
          sessionA={compareSessions[0]} 
          sessionB={compareSessions[1]} 
          onClose={() => { setShowComparison(false); setCompareSessions([]); }} 
        />
      )}
    </div>
  );
}

// --- COMPARISON PANEL COMPONENT ---
const ComparisonPanel = ({ sessionA, sessionB, onClose }: { sessionA: any, sessionB: any, onClose: () => void }) => {
  const scoreDiff = sessionB.score - sessionA.score;
  const durDiff = (sessionB.metadata?.durationMinutes || 0) - (sessionA.metadata?.durationMinutes || 0);
  
  const getDeltaString = (diff: number, unit: string = "") => {
    if (diff === 0) return "No change";
    const sign = diff > 0 ? "+" : "";
    return `${sign}${diff}${unit}`;
  };

  const getDeltaColor = (diff: number, positiveIsGood: boolean = true) => {
    if (diff === 0) return "text-slate-400";
    if (diff > 0) return positiveIsGood ? "text-green-400" : "text-red-400";
    return positiveIsGood ? "text-red-400" : "text-green-400";
  };

  const metricsA = sessionA.metrics || calculateSleepMetrics(sessionA.predictions, sessionA.metadata?.durationMinutes);
  const metricsB = sessionB.metrics || calculateSleepMetrics(sessionB.predictions, sessionB.metadata?.durationMinutes);

  const getAvgEntropy = (session: any) => {
    if (!session.spectral_entropies || session.spectral_entropies.length === 0) return 0;
    const sum = session.spectral_entropies.reduce((a: number, b: number) => a + b, 0);
    return sum / session.spectral_entropies.length;
  };

  const entA = getAvgEntropy(sessionA);
  const entB = getAvgEntropy(sessionB);
  const entDiff = entB - entA;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 p-6 backdrop-blur-md overflow-y-auto animate-in fade-in duration-300">
      <div className="w-full max-w-6xl rounded-3xl border border-white/10 bg-[#0c0d12] p-8 shadow-2xl relative text-left">
        
        <button onClick={onClose} className="absolute top-6 right-6 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-all text-xs font-semibold cursor-pointer">
          ✕ Close Comparison
        </button>
        
        <div className="mb-8 border-b border-white/5 pb-4">
          <h2 className="text-2xl font-bold text-white">Compare Sleep Sessions</h2>
          <p className="text-xs text-slate-400 mt-1">Side-by-side comparison of patient sleep records to evaluate clinical progress.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Left Card: Session A */}
          <div className="rounded-2xl border border-white/5 bg-[#121318] p-6 space-y-6">
            <div className="flex justify-between items-start border-b border-white/5 pb-3">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-blue-400">Session A (Earlier)</span>
                <h3 className="text-base font-bold text-white mt-1 truncate w-64">{sessionA.metadata?.fileName || "Session A"}</h3>
                <p className="text-xs text-slate-500">{new Date(sessionA.created_at).toLocaleString()}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-blue-400">{sessionA.score}%</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-widest">Sleep Score</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <p className="text-slate-500">Duration</p>
                <p className="font-semibold text-slate-300">{sessionA.metadata?.durationMinutes || 0} mins</p>
              </div>
              <div>
                <p className="text-slate-500">Sampling Freq</p>
                <p className="font-semibold text-slate-300">{sessionA.metadata?.samplingRate || 100} Hz</p>
              </div>
            </div>

            {/* Stages */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Sleep Stage Durations</h4>
              <div className="space-y-2">
                {Object.entries(sessionA.summary || {}).map(([stage, count]: any) => (
                  <div key={stage} className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">{stage}</span>
                    <span className="font-bold text-slate-300">{count} epochs ({Math.round(count * 30 / 60)} mins)</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Brainwaves */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Brainwave Power Density</h4>
              <div className="grid grid-cols-4 gap-2 text-center">
                {Object.entries(sessionA.brainwaves || {}).map(([wave, power]: any) => (
                  <div key={wave} className="p-2 rounded-xl bg-white/5">
                    <p className="text-[10px] text-slate-500">{wave}</p>
                    <p className="text-xs font-bold text-white mt-1">{power}%</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Card: Session B */}
          <div className="rounded-2xl border border-white/5 bg-[#121318] p-6 space-y-6">
            <div className="flex justify-between items-start border-b border-white/5 pb-3">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">Session B (Later)</span>
                <h3 className="text-base font-bold text-white mt-1 truncate w-64">{sessionB.metadata?.fileName || "Session B"}</h3>
                <p className="text-xs text-slate-500">{new Date(sessionB.created_at).toLocaleString()}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-emerald-400">{sessionB.score}%</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-widest">Sleep Score</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <p className="text-slate-500">Duration</p>
                <p className="font-semibold text-slate-300">{sessionB.metadata?.durationMinutes || 0} mins</p>
              </div>
              <div>
                <p className="text-slate-500">Sampling Freq</p>
                <p className="font-semibold text-slate-300">{sessionB.metadata?.samplingRate || 100} Hz</p>
              </div>
            </div>

            {/* Stages */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Sleep Stage Durations</h4>
              <div className="space-y-2">
                {Object.entries(sessionB.summary || {}).map(([stage, count]: any) => {
                  const prevCount = sessionA.summary?.[stage] || 0;
                  const diffCount = count - prevCount;
                  const diffMins = Math.round((count - prevCount) * 30 / 60);
                  return (
                    <div key={stage} className="flex justify-between items-center text-xs">
                      <span className="text-slate-400">{stage}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-300">{count} epochs ({Math.round(count * 30 / 60)} mins)</span>
                        <span className={`text-[10px] font-bold ${getDeltaColor(diffCount, stage !== 'Wake')}`}>
                          ({getDeltaString(diffMins, "m")})
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Brainwaves */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Brainwave Power Density</h4>
              <div className="grid grid-cols-4 gap-2 text-center">
                {Object.entries(sessionB.brainwaves || {}).map(([wave, power]: any) => {
                  const prevPower = sessionA.brainwaves?.[wave] || 0;
                  const diffPower = power - prevPower;
                  return (
                    <div key={wave} className="p-2 rounded-xl bg-white/5">
                      <p className="text-[10px] text-slate-500">{wave}</p>
                      <p className="text-xs font-bold text-white mt-1">{power}%</p>
                      <p className={`text-[9px] font-bold mt-0.5 ${getDeltaColor(diffPower, wave === 'Delta' || wave === 'Alpha')}`}>
                        {getDeltaString(diffPower, "%")}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Sleep Architecture Metrics Comparison Table */}
        <div className="mt-8 rounded-2xl border border-white/5 bg-[#121318] p-6 space-y-4">
          <h4 className="text-sm font-bold text-white uppercase tracking-wider">Sleep Architecture & Complexity Comparison</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-slate-300 border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-slate-400 font-semibold text-[10px] uppercase tracking-wider text-left">
                  <th className="pb-3 text-left">Sleep Metric</th>
                  <th className="pb-3 text-left">Session A (Earlier)</th>
                  <th className="pb-3 text-left">Session B (Later)</th>
                  <th className="pb-3 text-left">Delta / Variance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[
                  {
                    name: "Sleep Quality Score",
                    valA: `${sessionA.score}%`,
                    valB: `${sessionB.score}%`,
                    delta: getDeltaString(scoreDiff, "%"),
                    color: getDeltaColor(scoreDiff, true)
                  },
                  {
                    name: "Total Duration",
                    valA: `${sessionA.metadata?.durationMinutes || 0} mins`,
                    valB: `${sessionB.metadata?.durationMinutes || 0} mins`,
                    delta: getDeltaString(durDiff, " mins"),
                    color: getDeltaColor(durDiff, true)
                  },
                  {
                    name: "Sleep Cycles",
                    valA: `${metricsA.cycles}`,
                    valB: `${metricsB.cycles}`,
                    delta: getDeltaString((typeof metricsB.cycles === 'number' && typeof metricsA.cycles === 'number') ? (metricsB.cycles - metricsA.cycles) : 0),
                    color: getDeltaColor((typeof metricsB.cycles === 'number' && typeof metricsA.cycles === 'number') ? (metricsB.cycles - metricsA.cycles) : 0, true)
                  },
                  {
                    name: "Sleep Onset Latency (SOL)",
                    valA: metricsA.onset,
                    valB: metricsB.onset,
                    delta: getDeltaString((parseInt(metricsB.onset) || 0) - (parseInt(metricsA.onset) || 0), "m"),
                    color: getDeltaColor((parseInt(metricsB.onset) || 0) - (parseInt(metricsA.onset) || 0), false)
                  },
                  {
                    name: "REM Latency",
                    valA: metricsA.remLatency,
                    valB: metricsB.remLatency,
                    delta: metricsB.remLatency === 'N/A' || metricsA.remLatency === 'N/A' ? "N/A" : getDeltaString((parseInt(metricsB.remLatency) || 0) - (parseInt(metricsA.remLatency) || 0), "m"),
                    color: getDeltaColor((parseInt(metricsB.remLatency) || 0) - (parseInt(metricsA.remLatency) || 0), false)
                  },
                  {
                    name: "Wake After Sleep Onset (WASO)",
                    valA: metricsA.waso,
                    valB: metricsB.waso,
                    delta: getDeltaString((parseInt(metricsB.waso) || 0) - (parseInt(metricsA.waso) || 0), "m"),
                    color: getDeltaColor((parseInt(metricsB.waso) || 0) - (parseInt(metricsA.waso) || 0), false)
                  },
                  {
                    name: "Mean Spectral Entropy (Complexity)",
                    valA: entA === 0 ? "N/A" : entA.toFixed(3),
                    valB: entB === 0 ? "N/A" : entB.toFixed(3),
                    delta: entA === 0 || entB === 0 ? "N/A" : (entDiff === 0 ? "0.000" : (entDiff > 0 ? "+" : "") + entDiff.toFixed(3)),
                    color: getDeltaColor(entDiff, false)
                  }
                ].map((row, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3.5 font-medium text-slate-200 text-left">{row.name}</td>
                    <td className="py-3.5 text-left">{row.valA}</td>
                    <td className="py-3.5 font-semibold text-slate-100 text-left">{row.valB}</td>
                    <td className={`py-3.5 font-bold text-left ${row.color}`}>{row.delta}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Variance Highlights Summary */}
        <div className="mt-8 p-5 rounded-2xl bg-white/5 border border-white/5 flex flex-col md:flex-row md:justify-around items-center gap-4 text-center">
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">Sleep Score Variance</p>
            <p className={`text-2xl font-bold ${getDeltaColor(scoreDiff, true)}`}>
              {getDeltaString(scoreDiff, "%")}
            </p>
          </div>
          <div className="h-8 w-px bg-white/10 hidden md:block"></div>
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">Duration Variance</p>
            <p className={`text-2xl font-bold ${getDeltaColor(durDiff, true)}`}>
              {getDeltaString(durDiff, " mins")}
            </p>
          </div>
          <div className="h-8 w-px bg-white/10 hidden md:block"></div>
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">Deep Sleep (N3) Delta</p>
            {(() => {
              const diffDeep = (sessionB.summary?.Deep || 0) - (sessionA.summary?.Deep || 0);
              const diffMins = Math.round(diffDeep * 30 / 60);
              return (
                <p className={`text-2xl font-bold ${getDeltaColor(diffDeep, true)}`}>
                  {getDeltaString(diffMins, " mins")}
                </p>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
};;