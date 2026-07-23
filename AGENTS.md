# Agent Guidelines for Transcribe Interviews

This document provides instructions, constraints, and best practices for AI Coding Agents (such as opencode, GitHub Copilot, Cursor, etc.) working on this repository.

---

## 📂 Key Files to Reference

When working on this codebase, pay special attention to:
* **`README`**: Located at the repository root. Contains detailed pipeline design, setup notes, and manual running steps. Make sure to update it if any core behaviors or pipeline steps change.
* **`.env`**: Located at the repository root (ignored by git). This file defines the operational environment. Refer to the **Environment Specification** section below for required environment variables.
* **`utils.py`**: Contains helper functions such as `run_func_w_progbar` and file path construction. Any modular or utility refactoring should be integrated here.

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

### 3. Environment Variables & Specification (`.env`)
The `.env` file specifies operational modes, directories, model sizes, and credentials. It is a critical runtime control file.

The complete env specification includes:
* `DATA_DIR`: Base directory for video/audio and transcription files (e.g., `dummy_data` or `data`).
* `DATA_SUBDIR`: Subdirectory name for the current active interview (e.g., `conversation1` or `my_interview_subdir`).
* `WHISPER_MODEL`: The Whisper transcription model size to load. Options include `tiny`, `base`, `small`, `medium`, `large-v3-turbo` (use `tiny` or `base` for local agent testing).
* `HF_TOKEN`: Hugging Face User Access Token (needed for downloading the gated `pyannote/speaker-diarization-3.1` model).
* `LITELLM_API_KEY`: The API key for the LiteLLM proxy or endpoint.
* `LITELLM_API_BASE`: The base URL endpoint for LiteLLM completion service (e.g., `https://litellm.dreamlab.ucsb.edu`).
* `LITELLM_TEST_MODEL`: The model name to use when testing (e.g., `litellm_proxy/gemini-3-flash-preview`).
* `LITELLM_PROD_MODEL`: The model name to use in production (e.g., `litellm_proxy/gemini-3.1-pro-preview-customtools`).

Example `.env` configuration:
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

## 📢 Key Recommendations for Code Reusers

If you are cloning or forking this repository to reuse it for your own transcription workflows, please observe the following requirements and recommendations:

1. **Hugging Face Model Access (Diarization):**
   * This project uses `pyannote/speaker-diarization-3.1` which is a **gated model**.
   * You **must** create a Hugging Face account, visit the [model page](https://huggingface.co/pyannote/speaker-diarization-3.1), accept the terms of service, and generate a User Access Token with read permissions.
   * Provide this token as `HF_TOKEN` in your `.env` file.

2. **LiteLLM / API Provider Adaptation:**
   * The default setup points to the UCSB DREAM Lab's LiteLLM proxy (`https://litellm.dreamlab.ucsb.edu`), which is restricted to UCSB library staff.
   * **To reuse this code elsewhere:** Update `LITELLM_API_BASE` and `LITELLM_API_KEY` in `.env` to point to your own LiteLLM server, or adapt the python completion calls in `merge.py`, `cleanup.py`, and `final_cleanup.py` to call standard endpoints like OpenAI, Anthropic, or local Ollama instances directly.

3. **Required System Utilities (`ffmpeg`):**
   * Audio splitting and WAV conversions depend heavily on `ffmpeg`. You must install `ffmpeg` on your host operating system (e.g., via `brew install ffmpeg` on macOS, or `sudo apt install ffmpeg` on Ubuntu/Debian).

4. **Resource Constraints and Hardware:**
   * Diarization runs on PyTorch. If a GPU (CUDA or MPS/Apple Silicon) is available, Pyannote and Whisper will utilize it. On standard CPU-only machines, diarization of a 1-hour interview can take 30+ minutes.
   * Ensure your system has sufficient RAM/VRAM if running larger Whisper models (such as `large-v3-turbo`).

5. **File Naming & Directory Conventions:**
   * The pipeline scripts are highly opinionated about file naming. Ensure your raw input recording is named exactly `recording.mp4` and is placed in the designated folder configured in `.env` (`DATA_DIR/DATA_SUBDIR/recording.mp4`) before executing `./run_pipeline.sh`.

---

## 🔧 Target Areas for Development

If you are tasked with enhancing this repository, prioritize the following features:
* **Granular Checkpointing & Resiliency:** Update `utils.run_func_w_progbar` to check if output files already exist and skip processing them.
* **Concurrency:** Parallelize the LLM API calls in `merge.py`, `cleanup.py`, and `final_cleanup.py` using `ThreadPoolExecutor`.
* **API Flexibility:** Add fallbacks/support for direct OpenAI/Anthropic/Ollama APIs instead of assuming only the UCSB LiteLLM Proxy.
* **Error Handling:** Add graceful retry decorators and error-handling routines around the Whisper, Pyannote, and LiteLLM invocations.
