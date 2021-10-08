from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd

# 넥슨(메이플스토리) 자유게시판에서 리뷰글 뽑아오기
driver = webdriver.Chrome('./chromedriver.exe')
driver.implicitly_wait(3)

# review : 전체 리뷰글
# plus   : 페이지 당 리뷰글(10개)
review = []
plus = []

# 가장 많이 보였던 메이플 스토리에 대한 리뷰 수집
# 메이플 스토리 홈페이지의 게시판에서 리뷰 수집
driver.get('https://maplestory.nexon.com/Community/Free?page=1')

while len(review)<20000:
    plus = driver.find_elements_by_xpath('//*[@id="container"]/div/div[1]/div/ul/li/p/a/span[2]')

    # selenium 객체 리스트에 저장된 형태를 text로 변경 및 전체 리뷰 리스트에 추가
    for i in range(len(plus)):
        plus[i] = plus[i].text
        review.append(plus[i])
    
    # 한 페이지 내 리뷰 뽑은 후 다음 페이지로 이동
    driver.find_element_by_xpath('//*[@id="container"]/div/div[1]/div/div[2]/span[3]/a').click()

game_review = pd.DataFrame({'txt':review,'label':None})

secs = time.time()+32400
tm = time.localtime(secs)
time_log = '{0}-{1}-{2}-{3}-{4}-{5}_gameRV'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)


game_review.to_csv('GameRV/%s.csv' %time_log, encoding = 'utf-8-sig')