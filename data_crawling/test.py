# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:42:40 2019

@author: 1615055
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

start_url = "https://book.naver.com/search/search.nhn?publishStartDay=&publishEndDay=&categoryId=&serviceSm=advbook.basic&ic=service.summary&title=&author=&publisher=&isbn={}&toc=&subject=&cate1Depth=&cate2Depth=&cate3Depth=&cate4Depth=&publishStartYear=&publishStartMonth=&publishEndYear=&publishEndMonth=&x=21&y=10"

datapath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\"

dataset = pd.read_csv(os.path.join(datapath, "book_info_no_duplicate.csv"))
book_id = dataset.book_id.values.tolist()
ISBN = dataset.ISBN.values.tolist()

def parse(start_url, book_id, ISBN) :
    
    res = []
    for ids, isbn in zip(book_id, ISBN) :
        print(ids)
        temp = [
                ids,
                isbn, 
                ]
        temp.extend(parse_query(start_url.format(isbn)))
        res.append(temp)
    return res
        
    
def parse_query(url) :
    
    query_page = requests.get(url)
    query_soup = BeautifulSoup(query_page.content, 'html.parser')
    
    try :
        book_url = query_soup.select('a[class^="N=a:bls.title"]')[0]['href']
    except :
        print(url)
        return ['', '']
    
    book_page = requests.get(book_url)
    book_soup = BeautifulSoup(book_page.content, 'html.parser')
    
    temp = []
    
    desc = book_soup.select('div[id^="bookIntroContent"] p')
    if len(desc) > 0 :
        desc = desc[0].text
    else :
        desc = ''
        
    pub_desc = book_soup.select('div[id^="pubReviewContent"] p')
    if len(pub_desc) > 0 :
        pub_desc = pub_desc[0].text
    else :
        pub_desc = ''
        
    temp.append(desc)
    temp.append(pub_desc)

    return temp

res = parse(start_url, book_id, ISBN)
    
data = pd.DataFrame(res, columns = ['book_id', 'ISBN', 'Description', 'Publisher Desc'])

data.to_csv(os.path.join(datapath, "book_description.csv"), index=False)








    
    
    
    
    
    
    
    
