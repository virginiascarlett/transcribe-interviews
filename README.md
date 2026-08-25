# Zoom Interview Transcript Generator

This project takes a Zoom recording and generates a higher-quality transcript
than the one provided by Zoom itself. It uses a Whisper model for transcription
and adds in speaker diarization with pyannote. The final transcript is the
result of merging and cleaning up these outputs using a large language model.

This pipeline is a very bespoke thing made for my immediate purposes, but I'm
slowly working to make it more generically useful.

## VERY IMPORTANT notes

- **⚠️IMPORTANT⚠️: To protect confidential interviews and secret API keys, add
  these lines to your .gitignore:**

```bash
data/
.env
```

- You'll have to configure your own LLM provider. Currently, the pipeline
  accepts two: OpenAI and LiteLLM. The choice of provider just dictates the
  syntax of the functions used to submit your request to the LLM, so your API
  just needs to be compatible with one of those two. Or if you want to use
  something else, you can write a new module in the style of my existing modules
  my_litellm.py and my_openai.py.

## How It Works

1. Transcription with Whisper AI
   - Transcribes your recording to produce a timestamped transcript that is more
     accurate than Zoom's default output. (It will take any mp4.)

Example:

```
[0.00s - 3.80s] We wanted to share five ways to reflect and align with this
[3.80s - 5.80s] Qi, the life force energy of May.
[5.80s - 10.30s] So think of it as a time to embrace expansion and growth.
[10.30s - 15.80s] May has that rising fire Qi and that's going to support this idea of
[15.80s - 19.40s] expansion and spreading out and exploring and venturing.
```

2. Speaker diarization with pyannote
   - The MP4 file is converted to WAV format, and then
     [pyannote-audio](https://github.com/pyannote/pyannote-audio) is used to
     identify and label which speaker spoke at what time.
   - This step generates time stamps and speaker IDs (e.g., SPEAKER_00,
     SPEAKER_01), but no transcript.

Example:

```
[0.0s - 6.0s] SPEAKER_00
[6.0s - 32.0s] SPEAKER_01
[32.0s - 52.8s] SPEAKER_00
[53.1s - 71.0s] SPEAKER_01
[71.2s - 85.4s] SPEAKER_01
```

3. Merge and clean
   - An LLM merges the time-stamped transcript with the time-stamped speaker
     labels to produce a complete transcript.

Example:

```
SPEAKER_00: We wanted to share five ways to reflect and align with this Qi, the life force energy of May.
SPEAKER_01: So think of it as a time to embrace expansion and growth. May has that rising fire Qi and that's going to support this idea of expansion and spreading out and exploring and venturing. This is a time to act on ideas, nurture maybe what you've started in the spring time, allow your energy to gain clarity and strength just beyond those early spring beginnings.
SPEAKER_00: Another way that you can reflect and align with the Qi, if May is to invite joy into everyday life. So joy and passion are emotions associated with the fire element.
```

- After merging, the transcript undergoes two clean-up steps that improve the readability of the transcript.

Example of final output:

```
SPEAKER_00: We wanted to share five ways to reflect and align with this Qi, the life force energy of May.

SPEAKER_01: Think of it as a time to embrace expansion and growth. May has that rising fire Qi and that is going to support this idea of expansion, spreading out, exploring, and venturing. This is a time to act on ideas, nurture what you have started in the springtime, and allow your energy to gain clarity and strength just beyond those early spring beginnings.

SPEAKER_00: Another way that you can reflect and align with the Qi of May is to invite joy into everyday life. Joy and passion are emotions associated with the fire element.

```

These examples used lower-quality models than I would use in a real production
run, and they are still cleaner than the .vtt file you get from Zoom. This
pipeline does not suffer from the "context rot" problem you're likely to
encounter if you try to clean up transcripts using a chatbot.

## Setup Instructions

0. (Run once) Clone the repo and build the virtual environment.

[Clone the GitHub repo.](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

To build the virtual environment, run:

```bash
uv sync
```

1. Activate the virtual environment.

```bash
source .venv/bin/activate
```

2. Prepare environment variables and directory structure.

Create a file in the root directory (transcribe-interviews/) called .env. In
this file, write in the following, substituting your own variables: (don't
include the comments)

```bash
DATA_DIR=my_data_dir # where all the data live; ADD TO .gitignore!!!
WHISPER_TEST_MODEL=tiny # model for testing transcription
WHISPER_PROD_MODEL=large-v3-turbo # model for final transcription
HF_TOKEN=my_HF_token # Follow the instructions here: https://github.com/pyannote/pyannote-audio You need to sign some forms on hugging face to get a free token
LITELLM_API_KEY=01234568abcdefg
LITELLM_API_BASE=https://example.com
LITELLM_TEST_MODEL=gemini-3.1-flash-lite
LITELLM_PROD_MODEL=gemini-3.1-pro-preview-customtools
OPENAI_API_KEY=01234568abcdefg
OPENAI_API_BASE=https://example.com
OPENAI_TEST_MODEL=amazon-nova-pro
OPENAI_PROD_MODEL=claude-v4.6-sonnet
```

Obviously, you don't need to include for credentials for OpenAI if you'll only
be using LiteLLM, and vice versa.

You can test that your connection to your provider is working by executing the
corresponding module as a script, e.g. ./my_litellm.py.

If you want to test the whole pipeline with the provided dummy data, you can use
this variable:

```bash
DATA_DIR=dummy_data
```

And execute the pipeline with this command, from the project root directory:

```bash
./run_pipeline.py --test --name may_qi --provider litellm
```

When you're ready to use real data, create a directory called `data/` (make sure
this is in `.gitignore` if the data are confidential!!!). Put your mp4 recording
from Zoom in that subdirectory.

Here are the arguments:

- `-t`/`-p` or `--test`/`--prod`: you MUST include one of these to indicate
  whether to use the test or production models configured in your .env file.
- `--name`: the name of the recording. Must match the basename of the mp4 file.
  So if your recording is called interview_with_joe.mp4, use
  `--name interview_with_joe`.
- `--provider`: one of either litellm or openai, as described above.

This pipeline is SLOW. 🐌 Expect it to take at least half the length of the
recording, so for a 1-hour interview expect it to take at least half an hour.
I just haven't had the time to make the necessary performance upgrades.

The pipeline produces a hidden file with temporary files and folders, named
`.tmp_<recording_name>`. Delete this at the end of the run if you are happy with the
final transcript. I plan to refactor the code so that it will resume at the last
complete file if the pipeline is interrupted, but this functionality is not here
yet.
