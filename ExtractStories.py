import requests
from bs4 import BeautifulSoup
import time


def ExtractStories(category):

	txt = open("raw/" + str(category) + ".txt", "r")
	urls = txt.readlines()

	count = 0
	for i in range(len(urls)):
		print(count)
		page = requests.get(urls[i].strip())
		soup = BeautifulSoup(page.content , 'html.parser')
		
		texts = soup.find_all(class_="rtejustify")

		if len(texts) > 0:
			f = open("Dataset/"+str(category) +"/" + str(count) + ".txt", "w+")
			title = soup.find(id="page-title")
			f.write(str(title.get_text()).strip())
			f.write("\n\n")
			story = ""
			for item in texts:
				story = story + " " + str(item.get_text())
			f.write(story)
			count = count + 1
			f.close()
		
for item in ["corporate", "market", "movies", "politics", "technology", "sports"]:
	print(item)
	ExtractStories(item)
