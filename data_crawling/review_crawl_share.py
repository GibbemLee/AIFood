# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:47:01 2019

@author: 1615055
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:12:27 2019

@author: 1615055
"""

from selenium import webdriver
import pandas as pd
import time
import os

'''
*********수정해야할 코드 부분*********
'''

# 데이터세트 저장 path
datapath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\"
# 리뷰 데이터 저장할 path
savepath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\reviews\\"
# 웹드라이버 있는 path
webdriverpath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\chromedriver"

# 수정 X
dataset = pd.read_csv(os.path.join(datapath, "book_info_drop_dup_no_desc.csv"))
book_ids = dataset.book_id.unique()

# 어디서부터 어디까지 크롤링 할건지 
start_number = 3000
end_number = 6000

''' 수정 끝! '''


url = "https://www.aladin.co.kr/shop/wproduct.aspx?ItemId={}"
driver = webdriver.Chrome(webdriverpath)


def scrape_review(book_id) :
    
    URL = url.format(book_id)
    driver.get(URL)
    time.sleep(0.2)
    
    # scroll down
    for i in range(0, 50) :
        driver.execute_script("window.scrollTo(0, window.scrollY + 600)") 
        time.sleep(0.2)
    
    time.sleep(0.2)
    
    # 전체 100자평 리뷰자 보기
    comment_total_element = driver.find_element_by_xpath('//*[@id="tabTotal"]')
    driver.execute_script("arguments[0].scrollIntoView();", comment_total_element)
    driver.execute_script("window.scrollTo(0, window.scrollY - 200)")
    comment_total_element.click()
    
    # find MORE button and move the page
    try :
        comment_more_element = driver.find_element_by_xpath("//*[@id='divReviewPageMore']/div[1]/a")
        driver.execute_script("arguments[0].scrollIntoView();", comment_more_element)
        driver.execute_script("window.scrollTo(0, window.scrollY - 200)") 
    
        # while there are more reviews, continue clicking on MORE
        while (True) :
            try : 
                choose_account = driver.find_element_by_xpath("//*[@id='divReviewPageMore']/div[1]/a")
                choose_account.click()
            except :
                break
    except :
        pass
    
    
    # 전체 마이리뷰 리뷰자 보기 
    review_total_element = driver.find_element_by_xpath('//*[@id="tabMyReviewTotal"]')
    driver.execute_script("arguments[0].scrollIntoView();", review_total_element)
    driver.execute_script("window.scrollTo(0, window.scrollY - 200)")
    review_total_element.click()
    
    # find MORE button and move the page
    try :
        review_more_element = driver.find_element_by_css_selector('div[class="Ere_prod_mblog_box np_myreview"] div div[class="Ere_btn_more"] a')
        driver.execute_script("arguments[0].scrollIntoView();", review_more_element)
        driver.execute_script("window.scrollTo(0, window.scrollY - 200)") 
    
        # while there are more reviews, continue clicking on MORE
        while (True) :
            try : 
                choose_account = driver.find_element_by_css_selector('div[class="Ere_prod_mblog_box np_myreview"] div div[class="Ere_btn_more"] a')
                choose_account.click()
            except :
                break
    except :
        pass
    
    # 100자평 리뷰 크롤링
    element = driver.find_elements_by_css_selector('div[id^="CommentReviewList"] div ul div[class^="hundred_list"]')

    res = []

    for elem in element :
    
        rating_str = [star.get_attribute('src') for star in elem.find_elements_by_css_selector('div[class^="HL_star"] img')]
        rating = rating_str.count('https://image.aladin.co.kr/img/shop/2018/icon_star_on.png')
    
        left_chunk = elem.find_element_by_css_selector('div[class^="left"]')
    
        user_name = left_chunk.find_element_by_css_selector('a').text
        user_id = left_chunk.find_element_by_css_selector('a').get_attribute('href').replace("http://blog.aladin.co.kr/", '')

        date_added = left_chunk.find_element_by_css_selector('span').text
    
        review_texts = elem.find_elements_by_css_selector('span[id^="spnPaper"]')
        review_text = [r.text for r in review_texts]
        review_text = ' '.join(review_text).strip()
    
        res.append([book_id, user_id, user_name, rating, date_added, review_text])
        
    # 마이리뷰 크롤링 
    element = driver.find_elements_by_css_selector('div[class="Ere_prod_mblog_box np_myreview"] div div[class^="hundred_list"]')    
    
    for elem in element :
        
        rating_str = [star.get_attribute('src') for star in elem.find_elements_by_css_selector('div[class^="HL_star"] img')]
        rating = rating_str.count('https://image.aladin.co.kr/img/shop/2018/icon_star_on.png')
    
        left_chunk = elem.find_element_by_css_selector('div[class^="left"]')
    
        user_name = left_chunk.find_element_by_css_selector('a').text
        user_id = left_chunk.find_element_by_css_selector('a').get_attribute('href').replace("http://blog.aladin.co.kr/", '')

        date_added = left_chunk.find_element_by_css_selector('span').text
    
        review_box = elem.find_elements_by_css_selector('div[class="blog_list3"] ul li')[1]

        try :
            more_btn = review_box.find_element_by_css_selector('a')
            more_btn.click()
        except :
            pass
        
        review_text = elem.find_elements_by_css_selector('div[id^="paperAll"]')
        review_text = [r.text for r in review_text]
        review_text = ' '.join(review_text).strip()
    
        res.append([book_id, user_id, user_name, rating, date_added, review_text])
        
    return res
    



filename = "review_{}_to_{}.csv"

for start, end in zip(range(start_number, end_number, 100), range(start_number+100, end_number, 100)) :
    temp_ids = book_ids[start:end]
    review_data = []

    for book_id in temp_ids :
        try :
            res = scrape_review(book_id)
            print(book_id, " scraped")
            review_data.extend(res)
        except :
            print("error on book_id: ", book_id)
            log =  open(os.path.join(savepath, "error_list.txt"), 'a+')
            log.write(str(book_id))
            log.write("\n")
            log.close()
        time.sleep(0.1)
        
    temp_dataset = pd.DataFrame(review_data, columns=['book_id', 'user_id', 'user_name', 'rating', 'time', 'review_text'])
    temp_filename = filename.format(start, end)
    temp_dataset.to_csv(os.path.join(savepath, temp_filename), index=False)
       
        
    
    
    
    







