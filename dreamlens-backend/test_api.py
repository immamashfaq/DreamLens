import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

# Add backend directory to sys.path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock AsyncIOMotorClient before importing main.py to prevent connection failures if Mongo is offline
import motor.motor_asyncio
original_motor_client = motor.motor_asyncio.AsyncIOMotorClient

mock_collection = MagicMock()
mock_collection.insert_one = AsyncMock(return_value=MagicMock())
mock_collection.find = MagicMock(return_value=MagicMock(
    sort=MagicMock(return_value=MagicMock(
        limit=MagicMock(return_value=AsyncMock(
            to_list=AsyncMock(return_value=[
                {
                    "_id": "603d7b931e5f3c001c8e1e77",
                    "user_id": "test_user",
                    "score": 90,
                    "metadata": {"fileName": "test.edf", "durationMinutes": 480},
                    "summary": {"Wake": 50, "Light": 400, "Deep": 200, "REM": 150},
                    "brainwaves": {"Delta": 20, "Theta": 40, "Alpha": 30, "Beta": 10},
                    "created_at": "2026-05-25T00:00:00Z"
                }
            ])
        ))
    ))
))
mock_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

# Apply mock database client
motor.motor_asyncio.AsyncIOMotorClient = MagicMock(return_value=MagicMock(
    dreamlens_db=MagicMock(reports=mock_collection)
))

# Import FastAPI app and RAG variables
from main import app, KNOWLEDGE_BASE, vectorizer, kb_matrix

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "DreamLens Engine is running", "database": "Connected"}

def test_rag_knowledge_base():
    # Verify that knowledge base has been parsed and vectorized
    assert len(KNOWLEDGE_BASE) > 0
    assert vectorizer is not None
    assert kb_matrix is not None

def test_chat_rag_routing():
    # Mock the global Groq client completions to test LLM pipeline offline
    import main
    original_groq = main.groq_client
    
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Verified RAG Answer: Deep sleep (N3) is characterised by high amplitude delta waves."))]
    main.groq_client.chat.completions.create = MagicMock(return_value=mock_completion)
    
    response = client.post("/chat", json={"message": "tell me about delta waves", "sleep_context": None})
    assert response.status_code == 200
    assert "reply" in response.json()
    assert "delta waves" in response.json()["reply"].lower()
    
    # Restore original client
    main.groq_client = original_groq

# Restore original database client at cleanup
motor.motor_asyncio.AsyncIOMotorClient = original_motor_client
