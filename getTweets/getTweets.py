# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:46:59 2018

@author: Delta III
"""
import time
from twython import Twython, TwythonError
import ast
import math
APP_KEY = 'hEtApsLL5BqTL3vl8BgsjxORm'
APP_SECRET = '0ondkQCDbEfayzaJv4AN24JrhH4B3vrhWeO61t9iYjZvMUpKiY'
OAUTH_TOKEN= '110128312-ALystGurjWFznmbO21gM2d5VxEaA7vg6qaKzuPzM'
OAUTH_TOKEN_SECRET ='DDvzWMipTrhbuHKweUnfvlWZ38leNntFrYy7HSoD9w9xX'
twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN,OAUTH_TOKEN_SECRET)
#va contenir les tweets
buffer = []
#twitter.search(q="smart cities")
#va contenir l'ensemble des id à requêter
requete = " "

#point de depart si besoin de restart
laststart = 6700500
fname = 'C:/Users/Delta III/Downloads/iottweets20092016.tsv/iot-tweets-2009-2016.tsv'
f =open(fname, encoding="utf8")

for i in range(laststart):
    f.readline()
modul = 4
resultfile = 'resfinalbistrois' + str(modul)
res  =open(resultfile,"a+", encoding="utf8")
curseur = laststart
results = twitter.request("https://api.twitter.com/1.1/application/rate_limit_status.json")
remaining = results['resources']['statuses']['/statuses/lookup']['remaining']
read = " "
while (read != ""):
    if (remaining == 0):
        results = twitter.request("https://api.twitter.com/1.1/application/rate_limit_status.json")
        remaining = results['resources']['statuses']['/statuses/lookup']['remaining']
        reset = results['resources']['statuses']['/statuses/lookup']['reset']
        if (remaining == 0):
            time.sleep(reset - time.time()+30)
    else :
        requete = ""
        for i in range(100-1):
            read = f.readline()
            #on recupere l'id du tweet
            if read != "" :
                content = str(read).split("\t")[0]
                requete +=content + ","
        
        content = str(f.readline()).split("\t")[0]
        requete +=content
        # print(requete)
        buffer = twitter.lookup_status(id = requete);
        
        centtweet = ""
        for i in buffer :
            centtweet += str(i)+"\n"
        res.write(centtweet)
        curseur = curseur + 100
        remaining = remaining -1
    print("last start", curseur, "remaning", remaining)

#To convert a line reprenenting a dict : ast.literal_eval(didi)

# you may also want to remove whitespace characters like `\n` at the end of each line
#content = [str(x).split("\t")[0] for x in content] 
#f.readline()

#print (content[laststart])
f.close()
res.close()
"""
results = twitter.request("https://api.twitter.com/1.1/application/rate_limit_status.json")
remaining = results['resources']['statuses']['/statuses/lookup']['remaining']
reset = results['resources']['statuses']['/statuses/lookup']['reset']
results = twitter.lookup_status(id = "");
"""