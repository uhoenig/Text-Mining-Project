#Import links and keep the working ones using try-except function. At the end of this cell
#you have the uncleaned content of each article in a list: viral_content, non_viral
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time, random
import json
import os
import numpy as np
import pickle
from progressbar import ProgressBar, ReverseBar, ETA, Bar, Percentage, RotatingMarker,FileTransferSpeed
#os.chdir("/Users/uhoenig/Dropbox/Studium 2015-16 GSE/Term 3/Text Mining/Project/Data")

url_viral=pd.read_csv("pop5.csv").link #these are your functioning unviral links
url_nonviral=pd.read_csv("pop1.csv").link #these are your functioning viral links

#titles = [] maybe I extract titles later?

def checklinks(links,extension):
    #Predefined lists (necessary to have before loops)
    accepted_urls = []
    rejected_urls = []
    articles = []
    output = []
    idx = 0
    widgets = ['Test: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(links)).start()
    for link in links:

        delay = 0.0001*random.randint(0,1)
        #pause execution for 1-2 seconds
        time.sleep(delay)

        try:
            #open url and read content
            page = requests.get(link)
            soup = BeautifulSoup(page.content)

            #Put the functioning link into the accepted url list
            accepted_urls.append(link)

            #Store each "document" separately and extract the text from links
            temp = soup.find_all("section", {"class": "article-content"})[0].find_all('p')
            test = []
            for i in range(len(temp)):
                str(test.append(temp[i].text))
            articles.append(','.join(test))

        except:
            rejected_urls.append(link)
        #save results after every 10 iterations
        if np.mod(idx,100)==0:
            pickle.dump(articles,open("articles_"+extension+".pkl","w"))
            pickle.dump(rejected_urls,open("rejected_"+extension+".pkl","w"))
            pickle.dump(accepted_urls,open("accepted_"+extension+".pkl","w"))
            pickle.dump(idx,open("idx_"+extension+".pkl","w"))
        idx += 1
        pbar.update(idx)
    #save everything
    pickle.dump(articles,open("articles_"+extension+".pkl","w"))
    pickle.dump(rejected_urls,open("rejected_"+extension+".pkl","w"))
    pickle.dump(accepted_urls,open("accepted_"+extension+".pkl","w"))
    pickle.dump(idx,open("idx_"+extension+".pkl","w"))
    return (articles,accepted_urls, rejected_urls)

#Save them, so that you don't have to scrape them again

def save(name,file):
    with open(name, 'w') as texts:
        json.dump( file, texts, indent = 3 )
    print "Dumped File"

data_viral=checklinks(url_viral,"viral")
data_nonviral=checklinks(url_nonviral,"nonviral")

save("rejected_urls_viral.txt", data_viral[2])
save("rejected_urls_nonviral.txt", data_nonviral[2])
save("Documents_viral.txt",data_viral[0])
save("accepted_urls_viral.txt",data_viral[1])
save("Documents_nonviral.txt",data_nonviral[0])
save("accepted_urls_nonviral.txt",data_nonviral[1])
