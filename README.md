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
  just needs to be compatible with one of those two.

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

- After merging, the transcript undergoes two clean-up steps (again relying on
  LLMs):
  - `cleanup_step1.py` reformats the text for easier readability.
  - `cleanup_step2.py` removes filler words and corrects minor typos.

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
DATA_SUBDIR=my_interview_subdir # which interview you want to process
WHISPER_TEST_MODEL=tiny # model for testing transcription
WHISPER_PROD_MODEL=large-v3-turbo # model for final transcription
HF_TOKEN=my_HF_token # Follow the instructions here: https://github.com/pyannote/pyannote-audio You need to sign some forms on hugging face to get a free token
LITELLM_API_KEY=01234568abcdefg
LITELLM_API_BASE=https://example.com
LITELLM_TEST_MODEL=gemini-3.1-flash-lite
LITELLM_PROD_MODEL=gemini-3.1-pro-preview-customtools
OPENAI_API_KEY=01234568abcdefg
OPENAI_API_BASE=https://example.com
OPENAI_TEST_MODEL=claude-v4.5-haiku
OPENAI_PROD_MODEL=claude-v4.6-sonnet
```

Obviously, you don't need to include for credentials for OpenAI if you'll only
be using LiteLLM, and vice versa.

You can test that your connection to your provider is working by executing the
corresponding module as a script, e.g. ./my_litellm.py.

If you want to test the whole pipeline, you can use these variables:

```bash
DATA_DIR=dummy_data
DATA_SUBDIR=may_eq
```

That will point the pipeline to the included sample recording.

When you're ready to use real data, create a directory called `data/` (make sure
this is in `.gitignore` if the data are confidential!). In that directory,
create a subdirectory for the recording you want to process. Put your mp4
recording from Zoom in that subdirectory. **Rename the recording to
'recording.mp4'.**

3. Run the pipeline. **🛑 JK THIS SHELL SCRIPT IS CURRENTLY BROKEN, WILL FIX
   ASAP** Place the Zoom `.mp4` recording into a new subdirectory under `data/`.
   Here, I call it `my_interview_subdir/`. Make sure this name is in the .env
   file, above.

Next, you can run the whole pipeline all at once like so:

```bash
./run_pipeline.sh
```

N.B. this script has not been rigorously tested. In theory, it SHOULD resume at
the correct step if you've already completed a portion of the pipeline.

**All the following steps are optional. They show how to run the pipeline
manually, step by step.**

4. Run the following to split the file into several smaller files:

```bash
./split_recording.sh
```

5. The next two steps can be performed in any order:
   - Run `transcribe.py` on the subdirectory to produce a timestamped
     transcription with no speaker IDs:

     ```bash
     ./transcribe.py -t
     ```

     Use -t or --test to run in test mode with a your testing model, and -p or
     --prod to run in production mode with your production model. This step
     takes about 5 minutes for a one-hour interview.

   - Run these two commands to generate diarized timestamps with no transcript:
     ```bash
     ./convert_to_wav.sh
     ./diarize.py -t
     ```
     This step is SLOW. 🐌 I find that the required time is about half of the
     recording length. So diarize.py takes about 30 minutes for a 1-hour
     interview.

6. Run `./merge.py` to align the two transcripts. Example:

```bash
./merge.py -t --provider litellm
```

You must specify the provider, openai or litellm. This step takes about 15
minutes for a 1-hour interview.

7. Clean up the output in two steps:
   - `./cleanup_step1.py` – reformats for better readability.
   - `./cleanup_step2.py` – removes filler words and correct minor typos.

Again, specify test/prod and openai/litellmfor these, as above. Run with the -h flag, e.g.
`./cleanup_step1.py -h` to see the menu of options.
