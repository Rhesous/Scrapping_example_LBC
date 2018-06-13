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
    def __init__(self,request='',
                 del_prev_db=False,
                 dataloc='./datas/scrapped_lbc.db',
                 dataname='') :
        """
           Init the request for scrapping in LBC.

          """
        conn = sqlite3.connect(dataloc)
        cursor = conn.cursor()
        if del_prev_db==True:
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

        self.request=request
        self.del_prev_db=del_prev_db
        self.dataloc=dataloc
        self.dataname=dataname

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

    def update_db(self,step=10,nb_iter=100,time_sleep_=2,minrent_=500,maxrent_=1500,
                  msize_=0,maxsize_=15,minrooms_=3,maxrooms_=3):
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
        conn = sqlite3.connect(self.dataloc)
        cursor = conn.cursor()
        self.step=step
        self.nb_iter=nb_iter
        request_=urllib.request.quote(self.request)
        nb_page_=1
        i=1
        while(i<=self.nb_iter):
            lbc_page = ("https://www.leboncoin.fr/locations/offres/?o={}".format(nb_page_) +
                        "&q={}&location=Lyon&mrs={}&mre={}&sqs={}&sqe={}&ros={}&roe={}&ret=2".format(
                            request_,
                            minrent_,
                            maxrent_,
                            msize_,
                            maxsize_,
                            minrooms_,
                            maxrooms_))
            # Query the website and return the html to the variable 'page'
            page = urllib.request.urlopen(lbc_page)
            # Parse the html in the 'page' variable, and store it in Beautiful Soup format
            soup = BeautifulSoup(page, "lxml")

            for link in soup.find_all('a', attrs={'class': "list_item clearfix trackable"}):

                newlink = re.sub('&beta=1', '', 'https:' + link.get('href'))
                cursor.execute("""SELECT 1 FROM {} where link='{}'""".format(self.dataname,newlink))
                if (cursor.fetchone() is None):
                    offer=urllib.request.urlopen(newlink)
                    potage=BeautifulSoup(offer,"lxml")

                    # Title
                    try:
                        title = re.sub('\s\s+', ' ', link.find('p', attrs={'class': 'item_title'}).get_text())
                    except AttributeError:
                        title = re.sub('\s\s+', ' ', link.find('h2', attrs={'class': 'item_title'}).get_text())
                    except:
                        title = "TITLE NOT FOUND"

                    # Price
                    if len(link.find_all('p', attrs={'class': "item_price"})) > 0:
                        price = re.sub('\s\s+|\n+', ' ', link.find('p', attrs={'class': "item_price"}).get_text())
                    elif len(link.find_all('h3', attrs={'class': "item_price"})) > 0:
                        price = re.sub('\s\s+|\n+', ' ', link.find('h3', attrs={'class': "item_price"}).get_text())
                    else:
                        price = 'PRICE NOT FOUND'

                    # Import of divs to get refs
                    refs_ = potage.find_all('div',attrs={'data-reactid':re.compile('\D*')})

                    # Number of rooms
                    try :
                        id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Pièces"][0]) + 1
                        nb_rooms_ = potage.find('div',attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                        nb_rooms_ = re.sub('\s\s+|\n+', ' ', nb_rooms_)
                    except Exception:
                        nb_rooms_ = None

                    # Surface
                    try:
                        id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Surface"][0]) + 1
                        surface_ = potage.find('div',
                                               attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                        surface_ = re.sub('\s\s+|\n+', ' ', surface_)
                    except Exception:
                        surface_ = None

                    # Charges
                    try:
                        id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Charges comprises"][0]) + 1
                        charges_ = potage.find('div',
                                               attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                        charges_ = re.sub('\s\s+|\n+', ' ', charges_)
                    except Exception:
                        charges_ = None

                    # Charges
                    try:
                        id_ = int([x.get('data-reactid') for x in refs_ if x.getText() == "Meublé / Non meublé"][0]) + 1
                        furnished_ = potage.find('div',
                                               attrs={'class': "_3Jxf3", 'data-reactid': id_}).get_text()
                        furnished_ = re.sub('\s\s+|\n+', ' ', furnished_)
                    except Exception:
                        furnished_ = None

                    # Récupération de l'adresse
                    city=re.sub("Voir sur la carte","",potage.find('div', attrs={'class':'_1aCZv'}).get_text())

                    # Récupération du texte
                    description=re.sub('\s\s+',' ',potage.find('span',attrs={'class':"_2wB1z"}).get_text())

                    cursor.execute("""
                    INSERT INTO {} (link, title, price, city, nb_rooms,
                     surface, charges, furnished, description,update_date) 
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(self.dataname),
                                   (newlink,title,price,city, nb_rooms_, surface_, charges_, furnished_,
                                    description, datetime.datetime.now()))
                    if (i%step==0):
                        print('Committing {} new lines'.format(step))
                        evol_ = i/nb_iter*100
                        print("[" + "=" * int(evol_/2) + "-" * (50-int(evol_/2)) + "] {:.2f}%".format(evol_))
                        conn.commit()
                    i+=1
                    time.sleep(time_sleep_)

                if i-1 == self.nb_iter :
                    print('End of commits')
                    conn.commit()
                    conn.close()
                    return None
            nb_page_+=1
        conn.commit()
        conn.close()

if __name__ == '__main__':
    print("Hello world !")
    test = request_lb('T3', True, dataname='Lyon_rent')
    test.update_db(step=5,nb_iter=50,time_sleep_=0.05)

