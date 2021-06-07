import sklearn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
import pickle
from scipy import interp

from sklearn.svm import LinearSVC


data_dir = "/home/reddy/SAKSHI/Dataset/"

headlines = []
stories = []
labels = [] #

for category in os.listdir(data_dir):
	print(category)
	for item in os.listdir(data_dir + category):
		f = open(data_dir + category + "/" + str(item), "r")
		texts = f.readlines()
		temp = ""
		headlines.append(str(texts[0]))
		for i in range(1, len(texts)):
			temp = temp + " " + str(texts[i].strip())		
		stories.append(temp)

		if category == "economy":
			labels.append(0)
		elif category == "corporate":
			labels.append(1)
		elif category == "market":
			labels.append(2)  
		elif category == "movies":
			labels.append(3)
		elif category == "politics":
			labels.append(4)
		elif category == "technology":
			labels.append(5)
		elif category == "sports":
			labels.append(6)


target_names = ["economy", "corporate", "market", "movies", "politics", "technology", "sports"]

class_names = ["0", "1", "2", "3", "4", "5", "6"]


#Hash

vectorizer = HashingVectorizer(n_features=1000)
# encode document
vector = vectorizer.transform(headlines)
# summarize encoded vector
print(vector.shape)


X_train, X_test, y_train, y_test = train_test_split(vector, labels, test_size=0.2, shuffle=True, random_state=42)


clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf_lin = LinearSVC(random_state=0)
clf_lin.fit(X_train, y_train)
y_pred_lin_svm = clf_lin.predict(X_test)

clf_rd = RandomForestClassifier()
clf_rd.fit(X_train, y_train)
y_pred_rd = clf_rd.predict(X_test)

clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train, y_train)
y_pred_ada = clf.predict(X_test)

clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

clf_knn = KNeighborsClassifier(n_neighbors=7)
clf_knn.fit(X_train, y_train) 
y_pred_knn = clf_knn.predict(X_test)

clf_gnb = GaussianNB()
clf_gnb.fit(X_train.toarray(), y_train)
y_pred_gnb = clf_gnb.predict(X_test.toarray())

clf_bagging = BaggingClassifier(base_estimator=None)
clf_bagging.fit(X_train, y_train)
y_pred_bagging = clf_bagging.predict(X_test)


print("\n\n SVM\n")
print(classification_report(y_test, y_pred))#, class_names=class_names))

print("\n\n Linear SVM\n")
print(classification_report(y_test, y_pred_lin_svm))#, class_names=class_names))

print("\n\n Random Forest\n")
print(classification_report(y_test, y_pred_rd))#, class_names=class_names))

print("\n\n Ada Boost\n")
print(classification_report(y_test, y_pred_ada))#, class_names=class_names))

print("\n\n Decision Tree\n")
print(classification_report(y_test, y_pred_dt))#, class_names=class_names))

print("\n\n Logistic Regression\n")
print(classification_report(y_test, y_pred_lr))#, class_names=class_names))

print("\n\n KNN\n")
print(classification_report(y_test, y_pred_knn))#, class_names=class_names))

print("\n\n Gaussian NB\n")
print(classification_report(y_test, y_pred_gnb))#, class_names=class_names))

print("\n\n Bagging\n")
print(classification_report(y_test, y_pred_bagging))#, class_names=class_names))

"""

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
import seaborn as sn
import pandas as pd

matrix1 = confusion_matrix(y_test, y_pred)
matrix2 = confusion_matrix(y_test, y_pred_lin_svm)
matrix3 = confusion_matrix(y_test, y_pred_rd)
matrix4 = confusion_matrix(y_test, y_pred_ada)
matrix5 = confusion_matrix(y_test, y_pred_dt)
matrix6 = confusion_matrix(y_test, y_pred_lr)
matrix7 = confusion_matrix(y_test, y_pred_knn)
matrix8 = confusion_matrix(y_test, y_pred_gnb)
matrix9 = confusion_matrix(y_test, y_pred_bagging)



df_cm1 = pd.DataFrame(matrix1, index = class_names,
                  columns = class_names)
df_cm2 = pd.DataFrame(matrix2, index = class_names,
                  columns = class_names)
df_cm3 = pd.DataFrame(matrix3, index = class_names,
                  columns = class_names)
df_cm4 = pd.DataFrame(matrix4, index = class_names,
                  columns = class_names)
df_cm5 = pd.DataFrame(matrix5, index = class_names,
                  columns = class_names)
df_cm6 = pd.DataFrame(matrix6, index = class_names,
                  columns = class_names)
df_cm7 = pd.DataFrame(matrix7, index = class_names,
                  columns = class_names)
df_cm8 = pd.DataFrame(matrix8, index = class_names,
                  columns = class_names)
df_cm9 = pd.DataFrame(matrix9, index = class_names,
                  columns = class_names)


f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

#, (ax4, ax5, ax6),(ax7, ax8, ax9))

sn.heatmap(df_cm1, annot=True, ax=ax1).set_title('SVM')
sn.heatmap(df_cm2, annot=True, ax=ax2).set_title('Linear SVC')
sn.heatmap(df_cm3, annot=True, ax=ax3).set_title('Random Forest')
sn.heatmap(df_cm4, annot=True, ax=ax4).set_title('Ada Boost')
sn.heatmap(df_cm5, annot=True, ax=ax5).set_title('Decision Tree')
sn.heatmap(df_cm6, annot=True, ax=ax6).set_title('Logistic Regression')
sn.heatmap(df_cm7, annot=True, ax=ax7).set_title('KNN')
sn.heatmap(df_cm8, annot=True, ax=ax8).set_title('Gaussian NB')
sn.heatmap(df_cm9, annot=True, ax=ax9).set_title('Bagging')


#plt.tight_layout()
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig('CM.pdf')
#plt.savefig('CM.png')
plt.show()

"""
