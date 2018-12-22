import os
import requests as rq
import threading

from bs4 import BeautifulSoup

from matplotlib import pyplot as plt

POKE_URL = "https://pokemondb.net/pokedex/national"


class PokeSaver(threading.Thread):
    def __init__(self, num, name, type_l, url):
        super().__init__()
        self.num = num
        self.name = name
        self.type_l = type_l
        self.url = url

    def run(self):
        self.save_img()
        self.save_data()

    def save_img(self):
        print(f"Downloading {self.url}")
        img = rq.get(self.url).content
        with open(f"sprites/{self.num}.png", "wb") as file:
            file.write(img)

    def save_data(self):
        with open(f"data/{self.num}.txt", "w", encoding="utf8") as file:
            s = " ".join(self.type_l)
            file.write(f"{self.num} {self.name}\n{s}")


def fetch_all_pokemon():
    poke_page = rq.get(POKE_URL)
    soup = BeautifulSoup(poke_page.content, 'html.parser')
    thread_l = []
    for card in soup.find_all('div', {'class': 'infocard'}):
        num = int(card.find('small').text[1:])
        name = card.find('a', {'class': 'ent-name'}).text
        type_l = [a.text for a in card.find_all('a', {'class': 'itype'})]
        url = card.find('span', {'class': 'img-fixed'}).get('data-src')
        thread_l.append(PokeSaver(num, name, type_l, url))
        thread_l[-1].start()
    
    for t in thread_l:
        t.join()


def main():
    fetch_all_pokemon()
    pass


if __name__ == '__main__':
    main()
