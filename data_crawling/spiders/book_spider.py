# -*- coding: utf-8 -*-
"""
scrapy spider on mystery

Created on Tue Apr 30 15:12:07 2019

@author: 1615055
"""

import scrapy
import re

class BookSpider(scrapy.Spider) :
    
    name = 'book_info'

    def start_requests(self):
        
        start_url = "https://www.aladin.co.kr/shop/wbrowse.aspx?BrowseTarget=List&ViewRowsCount=50&ViewType=Detail&PublishMonth=0&SortOrder=4&page={}&Stockstatus=1&PublishDay=84&CID={}"
        max_id = 21
        CID = ["50926", # 미스테리
               "50927", # 라이트노벨
               "50928", # 판타지
               "50929", # 역사소설
               "50930", # SF
               "50931", # 호러, 공포
               "50932", # 무협
               "50933", # 액션, 스릴러
               "50935", # 로맨스
               "50948", # 희곡
               "50943", # 우리나라 옛글
               "51235", # 성장문학
               "51236", # 어른들을 위한 동화
               "51237", # 가족문학
               "51244", # 동물이야기
               "51253", # 전쟁문학
               "51252"  # 여성문학
               ]
        
        for c in CID :
            for i in range(1, max_id):
                yield scrapy.Request(start_url.format(str(i), c), callback=self.parse)
                
    
    def parse(self, response) :
        
        for result in response.css('a[class^="bo3"]::attr(href)').getall() :
            yield response.follow(result, callback=self.parse_book)
            
        
    def parse_book(self, response) :
            
        def extract_genres() :
            temp = response.css('ul[id^="ulCategory"] a::text').getall()
            if temp is None :
                return ''
            temp = list(set(temp))
            return temp
        
        def extract_ISBN() :
            temp = response.css('meta[property^="og:barcode"]').get(default='')
            isbn = re.findall(r'\d+', temp)[0]
            return isbn
        
        def extract_price() :
            temp = response.css('meta[property^="og:price"]').get(default='')
            price = re.findall(r'\d+', temp)[0]
            return price
        
        def extract_page_num() :
            temp = response.css('div[class^="conts_info_list1"] ul li::text').getall()
            if temp is None :
                return ''
            page_num = re.findall(r'\d+', temp[1])[0]
            return page_num
        
        def extract_book_id() :
            temp = response.css('meta[property^="og:url"]').get(default='')
            book_id = re.findall(r'\d+', temp)[0]
            return book_id
        
        def extract_author() :
            temp = response.css('li[class^="Ere_sub2_title"] a[href^="/search/wsearchresult.aspx?AuthorSearch"]::text').get(default='')
            return temp
            
        def extract_publisher() :
            temp = response.css('li[class^="Ere_sub2_title"] a[href^="/search/wsearchresult.aspx?PublisherSearch"]::text').get(default='')
            return temp
        
        def extract_title() :
            temp = response.css('div[class^="Ere_prod_titlewrap"] a[class^="Ere_bo_title"]::text').get(default='')
            return temp
        
        def extract_summary() :
            temp = response.css('div[class^="Ere_prod_mconts_box"]')
            contents = temp.getall()
            search = "<div class=\"Ere_prod_mconts_L\">책소개"
            indices = [i for i, s in enumerate(contents) if search in s]
            res = temp[indices[0]].css('div[class^="Ere_prod_mconts_R"]::text').getall()
            return res
        
        
        yield {
            'book_id'   : extract_book_id(),
            'ISBN'      : extract_ISBN(),
            'title'     : extract_title(),
            'author'    : extract_author(),
            'genre'     : extract_genres(),
            'publisher' : extract_publisher(),
            'page_num'  : extract_page_num(),
            'price'     : extract_price(),
            #'summary'   : extract_summary()
        }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    