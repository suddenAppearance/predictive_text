import re
import unicodedata

filename = '.txt'
with open(filename, 'r', encoding='utf8') as f:
    text = f.read()
text = unicodedata.normalize('NFD', text)
text = re.sub('[.,!?:;"\'«»—…]|\xa0|\xc2\xa0', '\n', text)
text = re.sub('\[|]', '', text)
text = re.sub(r',| -|- |– | –', ' ', text)
text = re.sub('\n–', '\n', text)
# text = re.sub(r'[_\[\]]', ' ', text)
text = re.sub(r'[\d]', '', text)
# text = re.sub(r'\xa0|\xc2\xa0', '\n', text)
text = re.sub('\n ', '\n', text)
text = re.sub('[A-z]', '', text)
# text = re.sub(r'', '\n', text)
# text = re.sub(r' [гдезлмнпртфхцчшщъыьэю] ', ' ', text)
text = re.sub(r'  +', ' ', text)
text = re.sub('\n\n+', '\n', text)

with open(filename, 'w', encoding='utf8') as f:
    f.write(text)
