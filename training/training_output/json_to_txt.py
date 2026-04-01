import json

with open('classes.json', 'r', encoding='utf-8') as f:
    clases = json.load(f)

with open('kanjis.txt', 'w', encoding='utf-8') as f:
    for kanji in clases:
        f.write(f"{kanji}\n")