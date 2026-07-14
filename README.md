# DreamLens — Transformer-Based EEG Sleep Stage Classification

DreamLens is an AI-powered clinical sleep analysis assistant focused on classifying EEG sleep stages using deep learning. The system classifies sleep into Wake, N1, N2, N3, and REM stages based on American Academy of Sleep Medicine (AASM) sleep-stage standards.

It features a Next.js physician portal dashboard, a FastAPI processing engine, a PyTorch Transformer neural network, and an AI sleep assistant powered by Gemini with RAG clinical context.

---

## Aim & Methodology

The aim of this project is to develop a lightweight, computationally efficient Transformer-based model that can classify EEG sleep stages accurately.

The workflow of the system is:
1. **Load EEG Recordings**: Import clinical `.edf` signals.
2. **Preprocess EEG Signals**: Filter and downsample raw waveforms.
3. **Segment EEG Data**: Segment continuous signals into 30-second epochs.
4. **Deep Learning Classification**: Predict sleep stages using the trained Transformer model.
5. **RAG-Guided Sleep Coach**: Provide context-aware medical explanations using a vector database of sleep guidelines.
6. **Clinical Dashboard**: Visualize hypnograms, brainwave powers, patient details, and session comparisons.

---

## Project Structure

```
DreamLens-Final/
├── dreamlens-backend/    # FastAPI Backend, PyTorch models, MongoDB schemas
└── frontend/             # Next.js frontend with TailwindCSS, Clerk, and Recharts
```

---

## Prerequisites

Before running the application on a new device, ensure you have the following installed:

1. **Python** (version `3.10` or `3.11` is recommended)
2. **Node.js** (version `18` or higher)
3. **MongoDB** (running locally on `mongodb://localhost:27017` or accessible via a remote URI)

---

## Step-by-Step Setup & Running Guide

### 1. Database Setup
Ensure that **MongoDB** is installed and running on your system.
* Default local URI: `mongodb://localhost:27017`
* If you use a custom port or cloud instance (Atlas), have your connection URI ready.

---

### 2. Backend Setup (`dreamlens-backend`)

Open a terminal window and navigate to the backend directory:
```bash
cd dreamlens-backend
```

#### A. Set up Python Virtual Environment
Create and activate a virtual environment:
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### B. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### C. Create `.env` Environment File
Create a `.env` file inside the `dreamlens-backend` directory with the following variables:
```env
# MongoDB Connection URL
MONGODB_URL=mongodb://localhost:27017

# LLM API Keys (Gemini is primary, Groq is fallback)
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

#### D. Start the Backend Server
```bash
uvicorn main:app --reload --port 8000
```
The backend API will start running at `http://127.0.0.1:8000`.

---

### 3. Frontend Setup (`frontend`)

Open a separate terminal window and navigate to the frontend directory:
```bash
cd frontend
```

#### A. Install Node Dependencies
```bash
npm install
```

#### B. Create `.env.local` Environment File
Create a `.env.local` file inside the `frontend` directory with the following variables:
```env
# Clerk Authentication Keys (Get these from your Clerk dashboard)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key

# Backend Connection URL
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

#### C. Start the Frontend Development Server
```bash
npm run dev
```
The Next.js client will start running at `http://localhost:3000`.

---

## Verifying the Setup

1. Open your browser and navigate to `http://localhost:3000`.
2. Sign in using the Clerk authentication portal.
3. Once logged in, upload an `.edf` EEG file to check the model analysis pipeline, view graphs, and interact with the AI assistant.
