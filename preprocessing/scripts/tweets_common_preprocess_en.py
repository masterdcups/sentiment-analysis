# to deal with encoding issues (non ascii characters)
import sys
import os
import json
import csv
from pprint import pprint
import time

# tweet preprocessing lib/config
import preprocessor as p

# p.OPT.NUMBER, p.OPT.HASHTAG
# (do process them induvidually in analysis phase if required)
p.set_options(p.OPT.URL, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.RESERVED, p.OPT.EMOJI )


# regex to remove punctuations.
import re
import string

# '#' not included to save the space.
punctuations = '«»--!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~..."”“ '
regex = re.compile('[%s]' % re.escape(punctuations))
regexb=re.compile('b[\'\"]')

# data : read and write directory paths
read_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data', 'raw', '')
write_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data', 'clean', '')


def preprocessTweets(filepath):

    """
        read the json data from the given filepath
        return a list of the json objects

        preprocess rules:

            remove URL's
            remove pucntuations (NOT hash tags '#')
            remove emoji's
            remove smiley's
            remove mentions's
            remove retweets (duplicates only) (store original tweet)
            remove RESERVE words: RT etc
    """

    # write processed tweets to a csv
    out_file_path = os.path.join( write_data_path, 'data.csv') #outfile path
    out_file = open( out_file_path, "a")  # out_file obj
    csv_writer  = csv.writer(out_file, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["id", "text", "date"])  #write header

    # process each tweet (memory efficient for large files)
    id_list = []
    with open(filepath, 'r') as fileobj:
		
        for line in fileobj:
            try:
                print("hello")
                # raw data is doubly encoded json, pass it twice through json.loads
                line =  json.loads(json.dumps(line))
                #for obj in line:
                tweet=line['text']
                print(tweet)
                message_id=line['id']
                print(tweet)
                    #created_at = time.strftime('%Y-%m-%d %H', time.strptime(line["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
                    # remove url, emoji's, smirley's, mentions (you can choose to retain mentions)
                    # refer Global variable p.set_options
                tweet = p.clean(tweet)         # remove urls, reserved, emoji, smiley, mention
                tweet = tweet.lower()          # lower
                tweet = re.sub(r'[^\w\s\d]','',tweet)
                    #tweet = regexb.sub('',tweet)  # remove quotes
                    #tweet = regex.sub('', tweet)   # remove punctuations
                print(tweet, message_id, type(tweet), type(message_id))
                csv_writer.writerow([message_id, tweet])
            except:
                print('error')

    # close outfile
    out_file.close()



# returns the absolute paths of files presnet in a directory
# ("json files only")
def getFilePaths(directory):
    """
       returns a list of full paths for all the json files
       present in the given directory
    """

    list_of_filepaths=[]
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.json') :
                list_of_filepaths.append( os.path.join(root, f))
                print(list_of_filepaths)
    return list_of_filepaths

files = getFilePaths(read_data_path)

# preprocess all the raw json files
# output to data/clean/data.csv file
for filepath in files:
    preprocessTweets(filepath)
