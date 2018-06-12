import sqlite3
import urllib.request
from bs4 import BeautifulSoup
import re
import datetime
import time

dataloc='./datas/scrapped_lbc.db'
conn = sqlite3.connect(dataloc)
cursor = conn.cursor()
step = 1
nb_iter = 20
request_ = urllib.request.quote("Shadow of the colossus")
nb_page_ = 1
i = 1
category = "consoles_jeux_video"
while (i <= nb_iter):
    lbc_page = "https://www.leboncoin.fr/recherche/?text={}&category=43&region=12&page={}".format(request_,nb_page_)
    # Query the website and return the html to the variable 'page'
    page = urllib.request.urlopen(lbc_page)
    # Parse the html in the 'page' variable, and store it in Beautiful Soup format
    soup = BeautifulSoup(page, "lxml")

    for link in soup.find_all('a', attrs={'class': "clearfix trackable"}):

        newlink = re.sub('&beta=1', '', 'https://www.leboncoin.fr' + link.get('href'))
        cursor.execute("""SELECT 1 FROM scrapped_lbc where link='{}'""".format(newlink))
        if (cursor.fetchone() is None):
            offer = urllib.request.urlopen(newlink)
            potage = BeautifulSoup(offer, "lxml")
            # Récupération du texte
            for part in potage.find_all('p', attrs={'itemprop': "description"}):
                description = re.sub('\s\s+', ' ', part.get_text())
            # Récupération du titre
            for part in potage.find_all('h1', attrs={'itemprop': "name"}):
                title = re.sub('\s\s+', ' ', part.get_text())
            # Récupération du prix, pas toujours présent
            if len(potage.find_all('h2', attrs={'itemprop': "price"})) > 0:
                for part in potage.find_all('h2', attrs={'itemprop': "price"}):
                    price = re.sub('\s\s+', ' ', part.get('content'))
            else:
                price = ''
            # Récupération de l'adresse
            for span in potage.find_all('span', attrs={'itemprop': 'address'}):
                city = re.sub('\s\s+', ' ', span.contents[0])
            cursor.execute("""
                   INSERT INTO scrapped_lbc(link, title, price, city, description,update_date) 
                   VALUES(?, ?, ?, ?, ?, ?)""", (newlink, title, price, city, description, datetime.datetime.now()))
            if (i % 5 == 0):
                print('Je commit {} nouvelles lignes'.format(step))
                conn.commit()
            i += 1
            time.sleep(time_sleep_)

        if i - 1 == nb_iter:
            print('Fin des itérations')
            conn.commit()
            conn.close()
            return None
    nb_page_ += 1
conn.commit()
conn.close()