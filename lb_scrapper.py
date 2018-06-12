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
                 category='',
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
             price INTERGER,
             city TEXT,
             description TEXT,
             update_date DATETIME
        )
        """.format(dataname))
        conn.commit()
        conn.close()

        self.request=request
        self.category=category
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
             price INTERGER,
             city TEXT,
             description TEXT,
             update DATE
        )
        """.format(self.dataname))
        conn.commit()
        conn.close()

    def update_db(self,step=10,nb_iter=100,time_sleep_=2):
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
            lbc_page="https://www.leboncoin.fr/{}/offres/ile_de_france/?o={}&q={}".format(self.category,nb_page_,request_)
            #Query the website and return the html to the variable 'page'
            page = urllib.request.urlopen(lbc_page)
            #Parse the html in the 'page' variable, and store it in Beautiful Soup format
            soup = BeautifulSoup(page,"lxml")

            for link in soup.find_all('a',attrs={'class':"list_item clearfix trackable"}):

                newlink=re.sub('&beta=1','','http:'+link.get('href'))
                cursor.execute("""SELECT 1 FROM scrapped_lbc where link='{}'""".format(newlink))
                if (cursor.fetchone() is None):
                    offer=urllib.request.urlopen(newlink)
                    potage=BeautifulSoup(offer,"lxml")
                    # Récupération du texte
                    for part in potage.find_all('p',attrs={'itemprop':"description"}):
                        description=re.sub('\s\s+',' ',part.get_text())
                    # Récupération du titre
                    for part in potage.find_all('h1',attrs={'itemprop':"name"}):
                        title=re.sub('\s\s+',' ',part.get_text())
                    #Récupération du prix, pas toujours présent
                    if len(potage.find_all('h2',attrs={'itemprop':"price"}))>0:
                        for part in potage.find_all('h2',attrs={'itemprop':"price"}):
                            price=re.sub('\s\s+',' ',part.get('content'))
                    else: price=''
                    # Récupération de l'adresse
                    for span in potage.find_all('span', attrs={'itemprop':'address'}):
                        city=re.sub('\s\s+',' ',span.contents[0])
                    cursor.execute("""
                    INSERT INTO scrapped_lbc(link, title, price, city, description,update_date) 
                    VALUES(?, ?, ?, ?, ?, ?)""", (newlink,title,price,city,description, datetime.datetime.now()))
                    if (i%5==0):
                        print('Je commit {} nouvelles lignes'.format(step))
                        conn.commit()
                    i+=1
                    time.sleep(time_sleep_)

                if i-1 == self.nb_iter :
                    print('Fin des itérations')
                    conn.commit()
                    conn.close()
                    return None
            nb_page_+=1
        conn.commit()
        conn.close()

if __name__ == '__main__':
    print("Hello world !")
