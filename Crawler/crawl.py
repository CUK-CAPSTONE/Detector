from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time
import os
import urllib.request

# check your driver

keyword = ["사슴", "고양이", "토끼", "곰", "여우", "꼬부기"]
image_cnt = 1000

URL = 'site url' #

for word in keyword:
    driver = webdriver.Chrome()
    save_dir = "user_dir" + word

    os.makedirs(save_dir, exist_ok=True) # make directory
    os.chdir(save_dir)

    driver.get(URL + word + "/")

    for c in range(0, 20): # scroll web site
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, 'html.parser') # get src
    image_info_list = soup.find_all('img', class_='spacing_noMargin__Q_PsJ MediaCard_image__ljFAl')
    image_and_name_list = []

    print(f'=== Start === word : {word} ')

    download_cnt = 0
    for i in range(len(image_info_list)): # get information
        print(image_info_list[i].attrs)
        if i == image_cnt:
            break
        if 'src' in image_info_list[i].attrs:
            save_image = image_info_list[i]['src']

            image_path = os.path.join(keyword.replace(' ', '_') + '_' + str(download_cnt) + '.jpg')
            image_and_name_list.append((save_image, image_path))
            download_cnt += 1

    for i in range(len(image_and_name_list)): # download
        urllib.request.urlretrieve(image_and_name_list[i][0], image_and_name_list[i][1])

    print(f'=== End === word : {word} ')

    driver.close()