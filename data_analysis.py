import os
import csv
import numpy as np

data_dir = "/home/reddy/SAKSHI/Dataset/"

headlines = []
stories = []

h1=[]
h2=[]
h3=[]
h4=[]
h5=[]
h6=[]
h7=[]

s1=[]
s2=[]
s3=[]
s4=[]
s5=[]
s6=[]
s7=[]



for category in os.listdir(data_dir):
	print(category)
	for item in os.listdir(data_dir + category):
		f = open(data_dir + category + "/" + str(item), "r")
		texts = f.readlines()
		temp = ""

		for i in range(1, len(texts)):
			temp = temp + " " + str(texts[i].strip())		

		if category == "economy":
			h1.append(len(str(texts[0]).split(" ")))
			s1.append(len(temp.split(" ")))

		elif category == "corporate":
			h2.append(len(str(texts[0]).split(" ")))
			s2.append(len(temp.split(" ")))

		elif category == "market":
			h3.append(len(str(texts[0]).split(" ")))
			s3.append(len(temp.split(" ")))

		elif category == "movies":
			h4.append(len(str(texts[0]).split(" ")))
			s4.append(len(temp.split(" ")))

		elif category == "politics":
			h5.append(len(str(texts[0]).split(" ")))
			s5.append(len(temp.split(" ")))

		elif category == "technology":
			h6.append(len(str(texts[0]).split(" ")))
			s6.append(len(temp.split(" ")))

		elif category == "sports":
			h7.append(len(str(texts[0]).split(" ")))
			s7.append(len(temp.split(" ")))

headlines = h1+h2+h3+h4+h5+h6+h7
stories = s1+s2+s3+s4+s5+s6+s7

headlines = np.asarray(headlines)
h1 = np.asarray(h1)
h2 = np.asarray(h2)
h3 = np.asarray(h3)
h4 = np.asarray(h4)
h5 = np.asarray(h5)
h6 = np.asarray(h6)
h7 = np.asarray(h7)

stories = np.asarray(stories)	
s1 = np.asarray(s1)	
s2 = np.asarray(s2)	
s3 = np.asarray(s3)
s4 = np.asarray(s4)	
s5 = np.asarray(s5)	
s6 = np.asarray(s6)	
s7 = np.asarray(s7)

fopen = open("dataAnalysis.csv", "w+")
writer = csv.writer(fopen, delimiter=',')

writer.writerow(["total", "economy", "corporate", "market", "movies" , "politics", "technology", "sports"])
writer.writerow([len(headlines), len(h1), len(h2), len(h3), len(h4), len(h5), len(h6), len(h7)])
writer.writerow([round(np.mean(headlines),2), round(np.mean(h1),2), round(np.mean(h2),2), round(np.mean(h3),2), round(np.mean(h4),2), round(np.mean(h5),2), round(np.mean(h6),2), round(np.mean(h7),2)])
writer.writerow([np.median(headlines),np.median(h1), np.median(h2), np.median(h3), np.median(h4), np.median(h5), np.median(h6), np.median(h7)])
writer.writerow([round(np.std(headlines),2), round(np.std(h1),2), round(np.std(h2),2), round(np.std(h3),2), round(np.std(h4),2), round(np.std(h5),2), round(np.std(h6),2), round(np.std(h7),2)])
writer.writerow([np.max(headlines), np.max(h1), np.max(h2), np.max(h3), np.max(h4), np.max(h5), np.max(h6), np.max(h7)])
writer.writerow([np.min(headlines), np.min(h1), np.min(h2), np.min(h3), np.min(h4), np.min(h5), np.min(h6), np.min(h7)])

writer.writerow([round(np.mean(stories),2), round(np.mean(s1),2), round(np.mean(s2),2), round(np.mean(s3),2), round(np.mean(s4),2), round(np.mean(s5),2), round(np.mean(s6),2), round(np.mean(s7),2)])
writer.writerow([np.median(stories), np.median(s1), np.median(s2), np.median(s3), np.median(s4), np.median(s5), np.median(s6), np.median(s7)])
writer.writerow([round(np.std(stories),2), round(np.std(s1),2), round(np.std(s2),2), round(np.std(s3),2), round(np.std(s4),2), round(np.std(s5),2), round(np.std(s6),2), round(np.std(s7),2)])
writer.writerow([np.max(stories), np.max(s1), np.max(s2), np.max(s3), np.max(s4), np.max(s5), np.max(s6), np.max(s7)])
writer.writerow([np.min(stories), np.min(s1), np.min(s2), np.min(s3), np.min(s4), np.min(s5), np.min(s6), np.min(s7)])

fopen.close()

"""
print("variant", "total", "economy", "corporate", "market", "movies" , "politics", "technology", "sports")
print("LENGTH", len(headlines), len(h1), len(h2), len(h3), len(h4), len(h5), len(h6), len(h7))
print("\nHeadlines Length")



print("\nMEAN: ", np.mean(headlines), np.mean(h1), np.mean(h2), np.mean(h3), np.mean(h4), np.mean(h5), np.mean(h6), np.mean(h7))
print("\nMEDIAN: ", np.median(headlines), np.median(h1), np.median(h2), np.median(h3), np.median(h4), np.median(h5), np.median(h6), np.median(h7))
print("\nSTD: ", np.std(headlines), np.std(h1), np.std(h2), np.std(h3), np.std(h4), np.std(h5), np.std(h6), np.std(h7)) 	
print("\nMAX: ", np.max(headlines), np.max(h1), np.max(h2), np.max(h3), np.max(h4), np.max(h5), np.max(h6), np.max(h7)) 
print("\nMIN: ", np.min(headlines), np.min(h1), np.min(h2), np.min(h3), np.min(h4), np.min(h5), np.min(h6), np.min(h7)) 		

print("\nStories Length")

print("\nMEAN: ", np.mean(stories), np.mean(s1), np.mean(s2), np.mean(s3), np.mean(s4), np.mean(s5), np.mean(s6), np.mean(s7))
print("\nMEDIAN: ", np.median(stories), np.median(s1), np.median(s2), np.median(s3), np.median(s4), np.median(s5), np.median(s6), np.median(s7))
print("\nSTD: ", np.std(stories), np.std(s1), np.std(s2), np.std(s3), np.std(s4), np.std(s5), np.std(s6), np.std(s7)) 	
print("\nMAX: ", np.max(stories), np.max(s1), np.max(s2), np.max(s3), np.max(s4), np.max(s5), np.max(s6), np.max(s7)) 
print("\nMIN: ", np.min(stories), np.min(s1), np.min(s2), np.min(s3), np.min(s4), np.min(s5), np.min(s6), np.min(s7)) 		
"""
