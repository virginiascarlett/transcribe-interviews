import whisper

model = whisper.load_model("base")
result = model.transcribe('dummy_data/conversation1/test_speech.mp3')

outF = open('dummy_data/conversation1/results.txt', 'w')
for segment in result['segments']:
    outF.write(segment['text'].strip() + '\n')
outF.close()