import json
import soundfile as sf
import os


with open('SSRstranscriptG.json') as f:

    in_dict = json.load(f)


for d in in_dict:
    if not os.path.exists(d):
        print(d)