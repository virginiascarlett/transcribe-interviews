#!/usr/bin/env python
import whisper

model = whisper.load_model("base")
result = model.transcribe("dummy_data/conversation1/test_speech.mp3")

# To generate the transcript without time stamps:
# outF = open('dummy_data/conversation1/transcript1.txt', 'w')
# for segment in result['segments']:
#     outF.write(segment['text'].strip() + '\n')
# outF.close()

# To include timestamps:
outF2 = open("dummy_data/conversation1/transcript.txt", "w")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()
    outF2.write(f"[{start:.2f}s - {end:.2f}s] {text}" + "\n")
outF2.close()
