# Agent Guidelines for Transcribe Interviews

This document provides instructions, constraints, and best practices for AI Coding Agents (such as opencode, GitHub Copilot, Cursor, etc.) working on this repository.

---

## 🏗️ Repository Architecture & Workflow

This project is a modular pipeline that converts Zoom interview recordings (MP4) into high-quality transcripts using OpenAI Whisper, Pyannote Speaker Diarization, and LiteLLM-based post-processing.

### Pipeline Steps and Mapping:
1. **Splitting (`split_recording.sh`):** Converts the source recording (`recording.mp4`) into 8-minute chunk MP4s.
2. **Transcription (`transcribe.py`):** Runs Whisper to generate timestamped text chunks (`transcripts/transcript*.txt`).
3. **WAV Conversion (`convert_to_wav.sh`):** Converts MP4 chunks to WAV chunks.
4. **Diarization (`diarize.py`):** Uses Pyannote to label who spoke when (`diarizations/diarization*.txt`).
5. **Merge (`merge.py`):** Combines the transcript & diarization timestamps via LiteLLM into raw transcripts (`diarized_transcripts_raw/merged*.txt`).
6. **Cleanup (`cleanup.py`):** Removes redundant speaker labels via LiteLLM (`diarized_transcripts_clean/chunk*.txt`).
7. **Final Cleanup (`final_cleanup.py`):** Removes filler words and corrects punctuation via LiteLLM (`diarized_transcripts_clean_final/final*.txt`).
8. **Concatenation (`concat_final_results.sh`):** Joins all clean chunks into the final transcript (`final_transcript.txt`).

All of the above are coordinated by **`run_pipeline.sh`**.

---

## 🛠️ Setup & Running Environment

### 1. Python Environment
The project relies on a virtual environment named `.venv`.
* Ensure you locate and use `.venv/bin/python` or run `source .venv/bin/activate` before executing scripts.

### 2. Dependencies
* System dependency: `ffmpeg` must be installed on the system to split and convert audio files.
* Python dependencies are listed in `requirements.txt` (or installed in the `.venv` directory).

### 3. Environment Variables (`.env`)
A `.env` file must be present at the repository root. Ensure the following variables are configured correctly:
```bash
DATA_DIR=dummy_data
DATA_SUBDIR=conversation1
WHISPER_MODEL=tiny
HF_TOKEN=my_HF_token
LITELLM_API_KEY=my_litellm_key
LITELLM_API_BASE=https://litellm.dreamlab.ucsb.edu
LITELLM_PROD_MODEL=litellm_proxy/gemini-3.1-pro-preview-customtools
```

---

## ⚠️ Safety, Integrity, and Cost Guardrails

1. **Do Not Commit Data or Secrets:**
   * Never commit contents under the `data/` directory or any `.env` files. Ensure they remain ignored by `.gitignore`.
2. **Resource & API Cost Safety:**
   * Pyannote Diarization and large Whisper models can be compute-intensive and slow on CPU environments.
   * For testing/development, **always use the `tiny` or `base` Whisper models** and dummy data (under `dummy_data/conversation1`). Do not invoke large production models unless explicitly requested.
3. **Rate Limits / LLM Calling:**
   * LiteLLM is used for the merging and clean-up steps. Be mindful of potential rate-limiting, and implement retries with exponential backoff if modifying the LLM calls.

---

## 🔧 Target Areas for Development

If you are tasked with enhancing this repository, prioritize the following features:
* **Granular Checkpointing & Resiliency:** Update `utils.run_func_w_progbar` to check if output files already exist and skip processing them.
* **Concurrency:** Parallelize the LLM API calls in `merge.py`, `cleanup.py`, and `final_cleanup.py` using `ThreadPoolExecutor`.
* **API Flexibility:** Add fallbacks/support for direct OpenAI/Anthropic/Ollama APIs instead of assuming only the UCSB LiteLLM Proxy.
* **Error Handling:** Add graceful retry decorators and error-handling routines around the Whisper, Pyannote, and LiteLLM invocations.
