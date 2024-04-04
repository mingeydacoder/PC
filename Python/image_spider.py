from selenium import webdriver
import time
import urllib
import os
# 存圖位置
local_path = 'D:'
# 爬取頁面網址 
url = 'https://pic.sogou.com/pics?query=%E4%B8%AD%E6%9D%91%E4%B8%80%E8%91%89&w=05009900'  

# 啟動chrome瀏覽器
driver = webdriver.Chrome()
driver.get("https://www.google.com/")

# 最大化窗口，因為每一次爬取只能看到視窗内的圖片  
driver.maximize_window()  

# 紀錄下載過的圖片網址，避免重複下載  
img_url_dic = {}  

# 瀏覽器打開爬取頁面
driver.get(url)  

# 模擬滾動視窗瀏覽更多圖片
pos = 0  
m = 0 # 圖片編號 
for i in range(10):  
    pos += i*500 # 每次下滾500  
    js = "document.documentElement.scrollTop=%d" % pos  
    driver.execute_script(js)  
    time.sleep(1)
    
    for element in driver.find_elements("xpath", '//*[@id="picPc"]/div/div/div/ul/li/div/a/img' ): #若改變搜尋目標xpath需修改
        try:
            img_url = element.get_attribute('src')
            
            # 保存圖片到指定路徑
            if img_url != None and not img_url in img_url_dic:
                img_url_dic[img_url] = ''  
                m += 1
                # print(img_url)
                ext = img_url.split('/')[-1]
                # print(ext)
                filename = str(m) + 'kazuha' + '_' + ext +'.jpg'
                print(filename)
                
                # 保存圖片
                urllib.request.urlretrieve(img_url, os.path.join(local_path , filename))
                
        except OSError:
            print('發生OSError!')
            print(pos)
            break;
            
driver.close()