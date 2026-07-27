# Zoom Interview Transcript Generator

This project takes a Zoom recording and generates a higher-quality transcript
than the one provided by Zoom itself. It uses a Whisper model for transcription
and adds in speaker diarization with pyannote. The final transcript is the
result of merging and cleaning up these outputs using a large language model via
LiteLLM.

It's pretty fragile at the moment, in that I don't have robust capabilities to
resume the process if it's interrupted, and it's dependent on LiteLLM as your
API gateway. This pipeline is a very bespoke thing made for my immediate
purposes, but I'm slowly working to make it more generically useful.

## Important notes

- This setup uses a LiteLLM service at the UCSB DREAM Lab that is restricted to
  library staff, but you could theoretically customize the scripts to use any
  LLM API.
- **To protect confidential interviews and secret API keys, add these lines to
  your .gitignore:**

```bash
data/
.env
```

## How It Works

1. Transcription with Whisper AI
   - Transcribes the Zoom audio (MP4) to produce a timestamped transcript that
     is more accurate than Zoom's default output.

2. Speaker diarization with pyannote
   - The MP4 file is converted to WAV format, and then
     [pyannote-audio](https://github.com/pyannote/pyannote-audio) is used to
     identify and label which speaker spoke at what time.
   - This step generates time stamps and speaker IDs (e.g., SPEAKER_00,
     SPEAKER_01), but no transcript.

3. Merge and clean
   - An LLM merges the time-stamped transcript with the time-stamped speaker
     labels to produce a complete transcript.
   - After merging, the transcript undergoes two clean-up steps (again relying
     on LLMs):
     - `cleanup.py` reformats the text for easier readability.
     - `final_cleanup.py` removes filler words and corrects minor typos.

## Setup Instructions

1. Activate the virtual environment.

```bash
source .venv/bin/activate
```

2. Prepare environment variables: in .env, write in the following:

```bash
DATA_DIR=my_data_dir
DATA_SUBDIR=my_interview_subdir
WHISPER_TEST_MODEL=tiny
WHISPER_PROD_MODEL=large-v3-turbo
HF_TOKEN=my_HF_token
LITELLM_API_KEY=my_litellm_key
LITELLM_API_BASE=https://litellm.dreamlab.ucsb.edu
LITELLM_TEST_MODEL=litellm_proxy/gemini-3-flash-preview
LITELLM_PROD_MODEL=litellm_proxy/gemini-3.1-pro-preview-customtools
```

If you want to test the pipeline, you can use these variables:

```bash
DATA_DIR=dummy_data
DATA_SUBDIR=conversation1
```

I use the 'tiny' Whisper model for testing and 'large-v3-turbo' for production.
Pyannote requires that you sign some forms on hugging face in order to get a
token to use it--that's where the HF token comes from.

The LiteLLM "API Gateway" for UCSB Library staff give us access to
"gemini-3-flash-preview", "gemini-3.1-pro-preview", or
"gemini-3.1-pro-preview-customtools". Flash is cheaper, pro is smarter.

3. Place the Zoom `.mp4` recording into a new subdirectory under `data/`. Here,
   I call it `my_interview_subdir/`. Make sure this name is in the .env file.

Next, you can either run the whole pipeline all at once like so:

```bash
./run_pipeline.sh
```

Or you can proceed with the step-by-step procedure below. N.B. The above shell
script has not been rigorously tested. In theory, it SHOULD resume at the
correct step if you've already completed a portion of the pipeline.

4. Rename the recording to `recording.mp4`.

5. Run the following to split the file into several smaller files:

```bash
./split_recording.sh
```

6. The next two steps can be performed in any order:
   - Run `transcribe.py` on the subdirectory to produce a timestamped
     transcription with no speaker IDs:

     ```bash
     ./transcribe.py
     ```

     This takes about 5 minutes for a one-hour interview.

   - Run these two commands to generate diarized timestamps with no transcript:
     ```bash
     ./convert_to_wav.sh
     ./diarize.py
     ```
     diarize.py takes about 30 minutes for a 1-hour interview.

7. Run `./merge.py` to align the two transcripts. This step takes about 15
   minutes for a 1-hour interview.

8. Clean up the output in two steps:
   - `./cleanup.py` – reformats for better readability.
   - `./final_cleanup.py` – uses AI to remove filler words and correct minor
     typos.
