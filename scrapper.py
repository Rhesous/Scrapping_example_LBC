import sys
import lb_scrapper

if __name__ == '__main__':
    try:
        request_target = sys.argv[1]
    except IndexError:
        request_target = "T3"
    try:
        delete_db = (sys.argv[2] == "True")
    except IndexError:
            delete_db = False
    try:
        db_name = sys.argv[3]
    except IndexError:
            db_name ="Lyon_rent"

    print("Hello world !")
    print("Options :\nRequest: {}\nDelete previous data: {}\nDataname: {}".format(request_target,delete_db,db_name))
    scapper = lb_scrapper.request_lb(request_target, delete_db, dataname=db_name)
    scapper.update_db(step=10, nb_iter=500, time_sleep_=0.05)
    scapper.check_nb_entries()
    print("Data collected thanks !")
