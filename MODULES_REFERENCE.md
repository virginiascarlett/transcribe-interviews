# Module Reference Guide

## Quick Start

### Running the Pipeline
```bash
python run_pipeline.py --name recording_name --prod --provider openai
python run_pipeline.py --name recording_name --test --provider litellm
```

### Individual Module Usage

#### Transcriber (transcriber.py)
Converts audio chunks to text transcripts.

**CLI:**
```bash
python transcriber.py --name recording_name --test
python transcriber.py --name recording_name --prod
```

**Module:**
```python
from transcriber import Transcriber, load_transcriber_config

data_dir, test_model, prod_model = load_transcriber_config()
transcriber = Transcriber(model_name=prod_model, data_dir=data_dir)
transcriber.transcribe_all("recording_name")
```

#### Diarizer (diarizer.py)
Identifies and labels different speakers in audio.

**CLI:**
```bash
python diarizer.py --name recording_name
```

**Module:**
```python
from diarizer import Diarizer, load_diarizer_config

data_dir, hf_token = load_diarizer_config()
diarizer = Diarizer(hf_token=hf_token, data_dir=data_dir)
diarizer.diarize_all("recording_name")
```

#### Merger (merger.py)
Combines transcript with speaker identification using an LLM.

**CLI:**
```bash
python merger.py --name recording_name --prod --provider openai
python merger.py --name recording_name --test --provider litellm
```

**Module:**
```python
from merger import Merger, load_merger_config

data_dir, test_model, prod_model, query_llm = load_merger_config("openai")
merger = Merger(llm_model=prod_model, data_dir=data_dir, query_llm_func=query_llm)
merger.merge_all("recording_name")
```

#### Cleaner (cleaner.py)
Two-step LLM-based transcript cleanup.

**CLI - Step 1 (Remove redundant speaker labels):**
```bash
python cleaner.py --name recording_name --step1 --prod --provider openai
python cleaner.py --name recording_name --step1 --test --provider litellm
```

**CLI - Step 2 (Grammar, punctuation, filler words):**
```bash
python cleaner.py --name recording_name --step2 --prod --provider openai
python cleaner.py --name recording_name --step2 --test --provider litellm
```

**Module:**
```python
from cleaner import Cleaner, load_cleaner_config

data_dir, test_model, prod_model, query_llm = load_cleaner_config("openai")
cleaner = Cleaner(llm_model=prod_model, data_dir=data_dir, query_llm_func=query_llm)

# First pass
cleaner.clean_step1("recording_name")

# Second pass
cleaner.clean_step2("recording_name")
```

## Backward Compatibility

Old script names still work (they forward to new modules):
- `transcribe.py` → `transcriber.py`
- `diarize.py` → `diarizer.py`
- `merge.py` → `merger.py`
- `cleanup_step1.py` → `cleaner.py --step1`
- `cleanup_step2.py` → `cleaner.py --step2`

## Key Configuration Functions

All modules provide a `load_*_config()` function that:
1. Loads environment variables from `.env`
2. Returns configuration in a tuple
3. Handles provider selection for LLM-based modules

**Available Functions:**
- `load_transcriber_config()` → (data_dir, whisper_test_model, whisper_prod_model)
- `load_diarizer_config()` → (data_dir, hf_token)
- `load_merger_config(provider)` → (data_dir, test_model, prod_model, query_llm_func)
- `load_cleaner_config(provider)` → (data_dir, test_model, prod_model, query_llm_func)

## Class Methods

### Transcriber
- `__init__(model_name, data_dir)` - Initialize with model and directory
- `transcribe_chunk(data_file)` - Transcribe a single file
- `transcribe_all(recording_name)` - Transcribe all chunks for a recording

### Diarizer
- `__init__(hf_token, data_dir)` - Initialize with HF token
- `diarize_chunk(data_file)` - Diarize a single file
- `diarize_all(recording_name)` - Diarize all chunks for a recording

### Merger
- `__init__(llm_model, data_dir, query_llm_func)` - Initialize with LLM settings
- `merge_chunk(diarization_file, transcript_file)` - Merge a single pair of files
- `merge_all(recording_name)` - Merge all chunks for a recording

### Cleaner
- `__init__(llm_model, data_dir, query_llm_func)` - Initialize with LLM settings
- `clean_chunk(data_file, step)` - Clean a single file (step=1 or step=2)
- `clean_step1(recording_name)` - First-pass cleanup
- `clean_step2(recording_name)` - Second-pass cleanup

## Environment Variables Required

See `.env` file for all required variables:
- `DATA_DIR` - Base directory for processing
- `WHISPER_TEST_MODEL` - Test Whisper model name
- `WHISPER_PROD_MODEL` - Production Whisper model name
- `HF_TOKEN` - Hugging Face API token
- `OPENAI_TEST_MODEL` - Test OpenAI model
- `OPENAI_PROD_MODEL` - Production OpenAI model
- `LITELLM_TEST_MODEL` - Test LiteLLM model
- `LITELLM_PROD_MODEL` - Production LiteLLM model
