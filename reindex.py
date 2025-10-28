#!/usr/bin/env python3
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "http://localhost:8000/reindex"
API_KEY = os.getenv("API_KEY", "your_secret_key")

headers = {"X-API-Key": API_KEY}

try:
    response = requests.post(API_URL, headers=headers)
    response.raise_for_status()
    print("✅ Reindex triggered successfully.")
    print("Response:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ Failed to trigger reindex:", e)