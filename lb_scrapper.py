import sqlite3
import urllib.request
from bs4 import BeautifulSoup
import re
import time
import datetime


def main_menu():
    return 0


class request_lb():
    """
       Class of scrapping request over leboncoin. This class consists in an object that can update the queries, delete
       current data and stock everything in a SQL database.
      :Attributes
            request:
                Input database in Termostat format
            category:
                Where to scrap in LBC
            dataloc:
                Where the database should be stocked
            dataname:
                Name of the database
            del_prev_db: Only used at init
                Restart with a new database if current already exists
      """

    def __init__(self, request='',
                 del_prev_db=False,
                 dataloc='./datas/scrapped_lbc.db',
                 dataname=''):
        """
           Init the request for scrapping in LBC.

          """
        conn = sqlite3.connect(dataloc)
        cursor = conn.cursor()
        if del_prev_db == True:
            cursor.execute("""
            DROP TABLE {}
            """.format(dataname))
            conn.commit()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS {}(
             link PRIMARY KEY UNIQUE,
             title TEXT,
             price INTEGER,
             city TEXT,
             nb_rooms INTEGER,
             surface INTEGER,
             charges TEXT,
             furnished TEXT,
             description TEXT,
             update_date DATETIME
        )
        """.format(dataname))
        conn.commit()
        conn.close()

        self.request = request
        self.del_prev_db = del_prev_db
        self.dataloc = dataloc
        self.dataname = dataname

    def check_nb_entries(self):
        """
           Check number of entries in the current SQL database.

          """
        conn = sqlite3.connect(self.dataloc)
        cursor = conn.cursor()
        sql = """SELECT count(*) as tot FROM {}""".format(self.dataname)
        cursor.execute(sql)
        print('Currently, {} entries.'.format(cursor.fetchone()[0]))
        conn.close()

    def delete_data(self):
        """
           Delete current database if needed.

          """
        conn = sqlite3.connect(self.dataloc)
        cursor = conn.cursor()
        cursor.execute("""
        DROP TABLE {}
        """.format(self.dataname))
        conn.commit()
        cursor.execute("""
        CREATE TABLE {}(
             link PRIMARY KEY UNIQUE,
             title TEXT,
             price INTEGER,
             city TEXT,
             nb_rooms INTEGER,
             surface INTEGER,
             charges TEXT,
             furnished TEXT,
             description TEXT,
             update_date DATETIME
        )
        """.format(self.dataname))
        conn.commit()
        conn.close()

    def update_db(self, step=10, nb_iter=100, time_sleep_=2, city_targ_="Lyon", minrent_=500, maxrent_=1700,
                  msize_=0, maxsize_=300, minrooms_=2, maxrooms_=5):
        """
           Update the scrapped database.
           :params
                pas: int
                    Rate at which we add a message to monitor the progress every
                nb_iter: int
                    Number of link to scrap
                time_sleep_: int
                    Waiting time between two calls

          """
        try:

            " Translation of categorical variable : size "
            conn = sqlite3.connect(self.dataloc)
            cursor = conn.cursor()
            self.step = step
            self.nb_iter = nb_iter
            request_ = urllib.request.quote(self.request)
            nb_page_ = 1
            i = 1
            while (i <= self.nb_iter):
                if request_ != "":
                    request_ = "?text={}&".format(request_)
                lbc_page = ("https://www.leboncoin.fr/recherche/"
                            + request_
                            + "?category=10&cities={}".format(city_targ_)
                            + "&real_estate_type=2&price={}-{}&rooms={}-{}&square={}-{}&page={}".format(
                            minrent_,
                            maxrent_,
                            minrooms_,
                            maxrooms_,
                            msize_,
                            maxsize_,
                            nb_page_
                        ))
                # Query the website and return the html to the variable 'page'
                page = urllib.request.urlopen(lbc_page)
                # Parse the html in the 'page' variable, and store it in Beautiful Soup format
                soup = BeautifulSoup(page, "lxml")
                # if len(soup.find_all("h1", attrs={'id': "result_ad_not_found_proaccount"})) > 0:
                if len(soup.find_all('p', attrs={'_2fdgs'})) > 0:
                    print("No more results to show")
                    break

                for link in soup.find_all('a', attrs={'class': "clearfix trackable"}):

                    newlink = re.sub('&beta=1', '', 'https://www.leboncoin.fr' + link.get('href'))
                    cursor.execute("""SELECT 1 FROM {} where link='{}'""".format(self.dataname, newlink))
                    if (cursor.fetchone() is None):
                        offer = urllib.request.urlopen(newlink)
                        potage = BeautifulSoup(offer, "lxml")

                        # Title
                        try:
                            title = re.sub('\s\s+', ' ', potage.find('h1').get_text())
                        except:
                            title = "TITLE NOT FOUND"

                        # Price
                        if len(link.find_all('span', attrs={'itemprop': "price"})) > 0:
                            price = re.search(r'\d+', (
                                link.find('span', attrs={'itemprop': "price"})
                                    .get_text()
                                    .strip()
                                    .replace(" ", "")
                            )).group(0)
                        elif len(link.find_all('h3', attrs={'class': "item_price"})) > 0:
                            price = re.search(r'\d+', (
                                link.find('h3', attrs={'class': "item_price"})
                                    .get_text()
                                    .strip()
                                    .replace(" ", "")
                            )).group(0)
                        else:
                            price = None

                        # Import of divs to get refs
                        refs_ = potage.find_all('div', attrs={'data-reactid': re.compile('\D*')})

                        # Number of rooms
                        try:
                            id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Pièces"][0]) + 1
                            nb_rooms_ = potage.find('div', attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                            nb_rooms_ = re.sub('\s\s+|\n+', ' ', nb_rooms_)
                        except Exception:
                            nb_rooms_ = None

                        # Surface
                        try:
                            id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Surface"][0]) + 1
                            surface_ = re.search(r'\d+', (
                                potage.find('div', attrs={'class': "_3Jxf3", 'data-reactid': id_})
                                    .get_text()
                                    .strip()
                                    .replace(" ", "")
                            )).group(0)
                            if surface_ == -1:
                                surface_ = None
                        except Exception:
                            surface_ = None

                        # Charges
                        try:
                            id_ = int(
                                [x.get('data-reactid') for x in refs_ if x.getText() == "Charges comprises"][0]) + 1
                            charges_ = potage.find('div',
                                                   attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                            charges_ = re.sub('\s\s+|\n+', ' ', charges_)
                        except Exception:
                            charges_ = None

                        # Charges
                        try:
                            id_ = int(
                                [x.get('data-reactid') for x in refs_ if x.getText() == "Meublé / Non meublé"][0]) + 1
                            furnished_ = potage.find('div',
                                                     attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                            furnished_ = re.sub('\s\s+|\n+', ' ', furnished_)
                        except Exception:
                            furnished_ = None

                        # Récupération de l'adresse
                        try:
                            city = re.sub("Voir sur la carte", "",
                                          potage.find('div',
                                                      attrs={'data-qa-id': 'adview_location_informations'}).get_text())
                        except:
                            city = city_targ_

                        # Récupération du texte
                        description = re.sub('<br>|<br/>|</br>', ' ',
                                             str(potage.find('span', attrs={'class': "_2wB1z"})))
                        description = re.search('<span .*>(.*?)<\/span>', description).group(1)

                        cursor.execute("""
                        INSERT INTO {} (link, title, price, city, nb_rooms,
                         surface, charges, furnished, description,update_date) 
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(self.dataname),
                                       (newlink, title, price, city, nb_rooms_, surface_, charges_, furnished_,
                                        description, datetime.datetime.now()))
                        if (i % step == 0):
                            print('Committing {} new lines'.format(step))
                            evol_ = i / nb_iter * 100
                            print("[" + "=" * int(evol_ / 2) + "-" * (50 - int(evol_ / 2)) + "] {:.2f}%".format(evol_))
                            conn.commit()
                        i += 1
                        time.sleep(time_sleep_)

                    if i - 1 == self.nb_iter:
                        print('End of commits')
                        conn.commit()
                        conn.close()
                        return None
                nb_page_ += 1
            conn.commit()
            conn.close()
        except KeyboardInterrupt:
            print("Update interrupted, you'll be able to continue it later !")
            print("For information, I was looking for :")
            print(newlink)
            print('At this page :')
            print(lbc_page)
            conn.commit()
            conn.close()
