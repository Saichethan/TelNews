import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

url = "https://www.sakshi.com/politics"
txt = open("raw/" + "politics.txt","w+")

def get_art():
	driver = webdriver.Chrome("/usr/bin/chromedriver")
	driver.get(url)

	for i in range(100):
		try:	
			driver.find_element_by_id("load_more").click()
			time.sleep(10)
		except:
			pass

	time.sleep(10)
	val = driver.find_elements_by_class_name("views-field-title")

	print(len(val))
	#print(val)

	for item in val:
		link = item.find_element_by_tag_name('a')
		href = link.get_attribute('href')
		txt.write(href)
		txt.write("\n")				

	driver.quit()
	txt.close()
	return val
	

t=get_art()
#for x in t:
#	print(x)




