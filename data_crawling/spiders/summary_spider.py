# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:46:38 2019

@author: 1615055
"""

import scrapy
import pandas as pd
import os

datapath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\"

class SummarySpider(scrapy.Spider) :
    
    name = 'summary'
    
    def start_requests(self):
        
        start_url = "https://www.aladin.co.kr/m/mproduct.aspx?ItemId={}"
        
        dataset = pd.read_csv(os.path.join(datapath, "book_info_no_duplicate.csv"))
        book_id = dataset.book_id.values.tolist()[:100]
        ISBN = dataset.ISBN.values.tolist()[:100]
        
        for ids, isbn in zip(book_id, ISBN) :
            yield scrapy.Request(start_url.format(ids), callback=self.parse_summary)
                    
            
    def parse_summary(self, response) :
        
        def extract_summary() :
            temp = response.xpath('//*[@id="introduce_short"]/text()').getall()
            if len(temp) > 0 :
                temp = [i.strip() for i in temp]
                res = ' '.join(temp)
            else :
                res = ''
            return res
        
        def extract_publ_sum() :
            temp = response.xpath('//*[@id="Publisher_short"]/text()').getall()
            if len(temp) > 0 :
                temp = [i.strip() for i in temp]
                res = ' '.join(temp)
            else :
                res = ''
            return res
        
        yield {"summary": extract_summary() }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        