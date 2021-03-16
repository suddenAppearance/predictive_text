import requests
import re
from bs4 import BeautifulSoup

with open('LIFE_wall.txt', 'w') as f:
    for i in range(0, 348549):
        try:
            content = requests.get(f'https://m.vk.com/wall-24199209_{16930256-i}').content
            content = content.decode('utf8')
            print(i)
            soup = BeautifulSoup(content, 'html.parser')
            element = soup.find('div', {'class': 'pi_text'})
            f.write(element.text + "\n")
            f.flush()
        except Exception as e:
            print(e)
