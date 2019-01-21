import ast

fname = 'C:/Users/Delta III/Downloads/iottweets20092016.tsv/iot-tweets-2009-2016.tsv'
#receuil =open(fname,"r", encoding="utf8")
#newfile = open("newfile02","w", encoding="utf8")
#Ajout : Text,List#, nblike, nbRT au csv original
nom_tri = "premiertritest00startnowserouslybisfinal"
last_start = 0
last_file = -1
receuil =open(fname,"r", encoding="utf8")
receuil.readline()
antidouble = set()


resfile = open(nom_tri,"a+", encoding="utf8")
number = "0"
testecriture = False
lines = receuil.readlines()
valeur = {}
count =0
new= ""
res= ""
count = 0
for l in lines:
    valeur [int(l.split("\t")[0])] = count
    count = count +1
print("Start try")
try:
    for modul in range(5):
        last_file = last_file +1
        
        resultfile = 'resfinalbistrois' + str(modul)
        result  =open(resultfile,"r", encoding="utf8")
        #receuil.close()
        #receuil =open(fname,"r", encoding="utf8")
        #read = receuil.readline()
        if False:
            pass
        else:
            dicta = []
            noms = ['text','id','favorite_count','retweet_count','entities']
            ligne = ' '
            while ligne != '':
                dicta = []
                for _ in range(50000):
                    ligne = result.readline()
                    if ligne != '':
                        temp = dict()
                        valtemp = ast.literal_eval(ligne)
                        for nom in noms:
                            temp[nom]=valtemp[nom]
                        dicta += [temp]
                for dic in dicta :
                    
                    last_start = last_start+1
                    testecriture = True
                    if (last_start % 1000 == 0):
                        print("Last start : ",last_start, "lastfile : ",last_file, "in ? ",dic['id'] in valeur)
                    try :
                        ind = valeur[dic['id']]
                        if ("|" not in dic['text'] and dic['id'] not in antidouble and  dic['in_reply_to_status_id'] == None): 
                            #ind = valeur.index(dic['id'])
                            content = lines[ind].split('\n')[0]
                            antidouble.add(int(dic['id']))
                            if ("|" not in content):
                                
            
                                hashtags = ""
                                for h in dic['entities']['hashtags'] :
                                    hashtags += "  " + h['text']
                                countcol = 0
                                for c in content.split('\t'):
                                    res += str(c) 
                                    countcol = countcol + 1
                                    if countcol>6:
                                        res += " "
                                    else :
                                        res += "|"
                                res += dic['text'].replace('\r', ' ').replace('\n', ' ') + "|"\
                                                            +hashtags + "|"\
                                                            +str(dic['favorite_count']) + "|"\
                                                            +str(dic['retweet_count'])\
                                                            +"\n"
                            
                            
                            testecriture = False
                            resfile.write(res)
                            #newfile.write(new)
                            res = ""
                            new= ""
                    except KeyError :
                        pass
except KeyboardInterrupt:
    print("Last start : ",last_start, "lastfile : ",last_file)

#newfile.write(new)
resfile.close()
#newfile.close()
