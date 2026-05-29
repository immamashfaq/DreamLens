import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import mne
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from mne.time_frequency import psd_array_welch
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# My custom model file
from models_architecture import DreamLens_Transformer

# --- 1. SETUP & KEYS ---
load_dotenv()

# Initialize the API
app = FastAPI(title="DreamLens Backend Engine")

# --- FIX: Initialize the Groq Client globally ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Allowing the React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:3001"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DB STUFF ---
print("🗄️ Connecting to MongoDB...")
MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
db_client = AsyncIOMotorClient(MONGO_URL)
database = db_client.dreamlens_db
reports_collection = database.reports
patients_collection = database.patients

@app.on_event("startup")
async def startup_db_client():
    try:
        await patients_collection.create_index("phone_number", unique=True)
        print("✅ MongoDB indexes initialized!")
    except Exception as e:
        print(f"⚠️ Error creating index: {e}")

# --- 3. LOADING THE MODEL ---
print("🧠 Waking up the DreamLens Transformer Engine...")
device = torch.device("cpu")
sleep_model = DreamLens_Transformer(num_classes=5, in_chans=1)

try:
    sleep_model.load_state_dict(torch.load("dreamlens_transformer_v1.pth", map_location=device, weights_only=True))
    sleep_model.eval() 
    print("✅ Transformer Model is ONLINE!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model weights. Error: {e}")

# --- 4. RAG KNOWLEDGE BASE INITIALIZATION ---
print("📖 Initializing RAG Knowledge Base...")
KNOWLEDGE_BASE = []
vectorizer = None
kb_matrix = None

kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleep_knowledge_base.txt")
if os.path.exists(kb_path):
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by lines, ignore empty lines and category headers
            lines = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith("[")]
            KNOWLEDGE_BASE = lines
        
        if KNOWLEDGE_BASE:
            vectorizer = TfidfVectorizer(stop_words='english')
            kb_matrix = vectorizer.fit_transform(KNOWLEDGE_BASE)
            print(f"✅ RAG Engine ONLINE! Loaded {len(KNOWLEDGE_BASE)} medical reference points.")
    except Exception as e:
        print(f"⚠️ Failed to load RAG knowledge base: {e}")
else:
    print(f"⚠️ RAG database file not found at: {kb_path}")

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    phone_number: str

class ChatRequest(BaseModel):
    message: str
    sleep_context: Optional[Dict[str, Any]] = None 

# --- 4. ROUTES ---

@app.get("/")
async def root():
    return {"status": "DreamLens Engine is running", "database": "Connected"}

@app.post("/analyze")
async def analyze_edf(
    file: UploadFile = File(...), 
    user_id: str = Form("guest_user"),
    patient_phone: str = Form(...),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None)
):
    print(f"📥 Processing: {file.filename} for Patient: {patient_phone}")
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
            
        raw = mne.io.read_raw_edf(temp_filename, preload=True, verbose=False)
        raw.pick(['EEG Fpz-Cz']) 
        raw.filter(0.3, 35.0, verbose=False)
        
        sfreq = raw.info['sfreq']
        duration_min = round((len(raw) / sfreq) / 60, 1)
        start_time = raw.info['meas_date'].strftime('%Y-%m-%d %H:%M:%S') if raw.info['meas_date'] else "Unknown"
        
        analysis_metadata = {
            "fileName": file.filename,
            "durationMinutes": duration_min,
            "startTime": start_time,
            "samplingRate": sfreq
        }

        nyquist = sfreq / 2
        potential_notches = [50, 100, 150]
        actual_notches = [f for f in potential_notches if f < nyquist - 2]
        if actual_notches:
            raw.notch_filter(actual_notches, verbose=False)

        data = raw.get_data()
        psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.5, fmax=30.0, n_per_seg=int(sfreq * 2), verbose=False)
        mean_psd = np.mean(psds, axis=0) 
        bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        band_powers = {}
        total_power = 0
        for band, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = np.sum(mean_psd[mask])
            band_powers[band] = power
            total_power += power
        
        brainwaves = {b: int(round((p / total_power) * 100)) for b, p in band_powers.items()} if total_power > 0 else {}

        epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True, verbose=False)
        X = epochs.get_data() * 1e6 
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)
        
        # Calculate Spectral Entropy for each epoch
        psds_epochs, freqs_epochs = psd_array_welch(X, sfreq=sfreq, fmin=0.5, fmax=30.0, n_per_seg=min(int(sfreq * 2), X.shape[-1]), verbose=False)
        spectral_entropies = []
        for i in range(len(X)):
            epoch_psd = psds_epochs[i, 0]
            psd_norm = epoch_psd / (np.sum(epoch_psd) + 1e-8)
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
            normalized_entropy = float(entropy / np.log2(len(psd_norm) + 1e-8))
            spectral_entropies.append(normalized_entropy)

        # Enable gradients for Class Saliency Maps (Explainable AI)
        X_tensor = torch.tensor(X.astype(np.float32)).to(device)
        X_tensor.requires_grad = True
        
        outputs = sleep_model(X_tensor)
        scores, preds = torch.max(outputs, 1)
        
        # Backprop predicted class score to calculate input gradients
        grad_outputs = torch.ones_like(scores)
        scores.backward(gradient=grad_outputs)
        saliency = X_tensor.grad.abs().cpu().numpy()
        X_tensor.grad = None # free gradients
        
        # Downsample signals and saliencies to 100 points for frontend rendering
        downsampled_signals = []
        downsampled_saliencies = []
        for i in range(len(X)):
            sig = X[i, 0]
            sal = saliency[i, 0]
            
            # Bulletproof downsampling to 100 bins using array_split
            sig_down = np.array([np.mean(chunk) for chunk in np.array_split(sig, 100)])
            sal_down = np.array([np.mean(chunk) for chunk in np.array_split(sal, 100)])
            
            # Normalize saliency to [0, 1] range for visual styling
            sal_min, sal_max = sal_down.min(), sal_down.max()
            if sal_max > sal_min:
                sal_down = (sal_down - sal_min) / (sal_max - sal_min)
            else:
                sal_down = np.zeros_like(sal_down)
                
            downsampled_signals.append(sig_down.tolist())
            downsampled_saliencies.append(sal_down.tolist())

        stage_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
        predicted_stages = [stage_map[int(p)] for p in preds]
        
        summary = {
            "Wake": predicted_stages.count("W"),
            "Light": predicted_stages.count("N1") + predicted_stages.count("N2"),
            "Deep": predicted_stages.count("N3"),
            "REM": predicted_stages.count("REM")
        }
        
        try:
            name_to_save = patient_name
            age_to_save = patient_age
            gender_to_save = patient_gender

            # Look up patient to get details if not provided
            p = await patients_collection.find_one({"phone_number": patient_phone})
            if p:
                if not name_to_save:
                    name_to_save = p.get("name")
                if age_to_save is None:
                    age_to_save = p.get("age")
                if not gender_to_save:
                    gender_to_save = p.get("gender")
            elif patient_name and patient_age is not None and patient_gender:
                # On-the-fly patient registration if details provided and patient not exists
                await patients_collection.update_one(
                    {"phone_number": patient_phone},
                    {"$setOnInsert": {
                        "name": patient_name,
                        "age": int(patient_age),
                        "gender": patient_gender,
                        "created_at": datetime.now(timezone.utc)
                    }},
                    upsert=True
                )
            
            report_doc = { 
                "user_id": user_id, 
                "patient_phone": patient_phone,
                "patient_name": name_to_save or "Unknown Patient",
                "patient_age": age_to_save,
                "patient_gender": gender_to_save or "Unknown",
                "metadata": analysis_metadata, 
                "summary": summary, 
                "brainwaves": brainwaves, 
                "predictions": predicted_stages,
                "spectral_entropies": spectral_entropies,
                "signals": downsampled_signals,
                "saliencies": downsampled_saliencies,
                "created_at": datetime.now(timezone.utc), 
                "score": 100 - int((summary["Wake"] / max(1, (summary["Wake"] + summary["Light"] + summary["Deep"] + summary["REM"]))) * 100) 
            }
            await reports_collection.insert_one(report_doc)
            print(f"✅ Successfully saved report to database for patient: {patient_phone}")
        except Exception as db_err:
            print(f"⚠️ DB Error: {db_err}")
            name_to_save = patient_name
            age_to_save = patient_age
            gender_to_save = patient_gender

        return {
            "status": "success",
            "metadata": analysis_metadata, 
            "summary": summary, 
            "brainwaves": brainwaves, 
            "predictions": predicted_stages,
            "spectral_entropies": spectral_entropies,
            "signals": downsampled_signals,
            "saliencies": downsampled_saliencies,
            "patient_phone": patient_phone,
            "patient_name": name_to_save or "Unknown Patient",
            "patient_age": age_to_save,
            "patient_gender": gender_to_save or "Unknown"
        }
        
    except Exception as e:
        print(f"❌ Error in ML Pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    try:
        cursor = reports_collection.find({"user_id": user_id}).sort("created_at", -1).limit(10)
        reports = await cursor.to_list(length=10)
        for report in reports:
            report["_id"] = str(report["_id"])
        return {"status": "success", "data": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{report_id}")
async def delete_report(report_id: str):
    try:
        result = await reports_collection.delete_one({"_id": ObjectId(report_id)})
        if result.deleted_count == 1:
            return {"status": "success", "message": "Report deleted"}
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        print(f"❌ Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat")
async def chat_with_groq(req: ChatRequest):
    try:
        # Search the local sleep knowledge base (RAG)
        context_sentences = []
        if vectorizer is not None and kb_matrix is not None and req.message:
            query_vec = vectorizer.transform([req.message])
            similarity = cosine_similarity(query_vec, kb_matrix).flatten()
            # Get top 3 matching facts
            top_indices = similarity.argsort()[-3:][::-1]
            for idx in top_indices:
                if similarity[idx] > 0.05: # threshold relevance
                    context_sentences.append(KNOWLEDGE_BASE[idx])

        sys_prompt = "You are the DreamLens AI assistant, an expert in sleep health. Keep answers concise, empathetic, and clinical."
        
        # Inject matching medical context
        if context_sentences:
            sys_prompt += "\nUse the following verified sleep science references to guide your answer:\n"
            sys_prompt += "\n".join([f"- {fact}" for fact in context_sentences])
            sys_prompt += "\nAlways base your advice on these clinical guidelines, and explicitly mention that this is for educational purposes."

        if req.sleep_context:
            clean_context = {k: v for k, v in req.sleep_context.items() if k not in ["signals", "saliencies", "predictions", "spectral_entropies"]}
            sys_prompt += f" The user's latest sleep data is: {clean_context}."

        # Check if Gemini key is available
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            import httpx
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "systemInstruction": {
                    "parts": [{"text": sys_prompt}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": req.message}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7
                }
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    res_data = res.json()
                    ai_response = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"reply": ai_response}
                else:
                    print(f"⚠️ Gemini API failed with status {res.status_code}: {res.text}")

        # Fallback to Groq if Gemini key is not configured or fails
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": req.message}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
            )
            ai_response = chat_completion.choices[0].message.content
            return {"reply": ai_response}

        raise Exception("No active AI API Key configured (neither GEMINI_API_KEY nor GROQ_API_KEY).")
        
    except Exception as e:
        print(f"❌ Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"The AI is currently asleep (Error: {str(e)}). Try again later!")

class SaveReportRequest(BaseModel):
    user_id: str
    patient_phone: str
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    brainwaves: Dict[str, Any]
    predictions: list[str]
    spectral_entropies: list[float]
    signals: list[list[float]]
    saliencies: list[list[float]]
    score: int

@app.post("/history")
async def save_report_to_history(req: SaveReportRequest):
    try:
        name_to_save = req.patient_name
        age_to_save = req.patient_age
        gender_to_save = req.patient_gender

        # Look up patient details in database if not provided
        p = await patients_collection.find_one({"phone_number": req.patient_phone})
        if p:
            if not name_to_save:
                name_to_save = p.get("name")
            if age_to_save is None:
                age_to_save = p.get("age")
            if not gender_to_save:
                gender_to_save = p.get("gender")
        elif req.patient_name and req.patient_age is not None and req.patient_gender:
            # On-the-fly patient registration if not exists
            await patients_collection.update_one(
                {"phone_number": req.patient_phone},
                {"$setOnInsert": {
                    "name": req.patient_name,
                    "age": req.patient_age,
                    "gender": req.patient_gender,
                    "created_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )

        report_doc = {
            "user_id": req.user_id,
            "patient_phone": req.patient_phone,
            "patient_name": name_to_save or "Unknown Patient",
            "patient_age": age_to_save,
            "patient_gender": gender_to_save or "Unknown",
            "metadata": req.metadata,
            "summary": req.summary,
            "brainwaves": req.brainwaves,
            "predictions": req.predictions,
            "spectral_entropies": req.spectral_entropies,
            "signals": req.signals,
            "saliencies": req.saliencies,
            "score": req.score,
            "created_at": datetime.now(timezone.utc)
        }
        result = await reports_collection.insert_one(report_doc)
        return {"status": "success", "message": "Report saved to history", "id": str(result.inserted_id)}
    except Exception as e:
        print(f"❌ Save Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW PATIENT PORTAL ROUTES ---

@app.get("/patients")
async def get_patients():
    try:
        cursor = patients_collection.find({}).sort("name", 1)
        patients = await cursor.to_list(length=100)
        for p in patients:
            p["_id"] = str(p["_id"])
            if "created_at" in p and isinstance(p["created_at"], datetime):
                p["created_at"] = p["created_at"].isoformat()
        return {"status": "success", "data": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{phone_number}")
async def get_patient(phone_number: str):
    try:
        p = await patients_collection.find_one({"phone_number": phone_number})
        if not p:
            raise HTTPException(status_code=404, detail="Patient not found")
        p["_id"] = str(p["_id"])
        if "created_at" in p and isinstance(p["created_at"], datetime):
            p["created_at"] = p["created_at"].isoformat()
        return {"status": "success", "data": p}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patients")
async def register_patient(req: PatientCreate):
    try:
        existing = await patients_collection.find_one({"phone_number": req.phone_number})
        if existing:
            raise HTTPException(status_code=400, detail="Patient with this phone number already registered")
        doc = {
            "name": req.name,
            "age": req.age,
            "gender": req.gender,
            "phone_number": req.phone_number,
            "created_at": datetime.now(timezone.utc)
        }
        await patients_collection.insert_one(doc)
        return {"status": "success", "message": "Patient registered successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{phone_number}/history")
async def get_patient_history(phone_number: str):
    try:
        cursor = reports_collection.find({"patient_phone": phone_number}).sort("created_at", -1)
        reports = await cursor.to_list(length=100)
        for report in reports:
            report["_id"] = str(report["_id"])
            if "created_at" in report and isinstance(report["created_at"], datetime):
                report["created_at"] = report["created_at"].isoformat()
        return {"status": "success", "data": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))