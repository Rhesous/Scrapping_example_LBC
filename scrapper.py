import lb_scrapper

mscrap=lb_scrapper.request_lb()
mscrap.update_db()

test=request_lb(request="Shadow of the Colossus", del_prev_db=True, category="consoles_jeux_video",dataname="sample")