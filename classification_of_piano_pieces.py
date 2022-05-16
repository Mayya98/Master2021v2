import pandas as pd
import os
from sklearn.utils import shuffle

def create_elementary_set(elementary_path):
    raw = 0
    column = ['name','text', 'level']
    piano_dataset1 = pd.DataFrame(columns=column)

    for folder in os.walk(elementary_path):
      for file in folder[2]:
        read_path = folder[0] + '/' + file
        with open(read_path, 'r') as f:
          text = f.read()
          file_name = file[:-4]
          piano_dataset1.at[raw, "name"] = file_name
          piano_dataset1.at[raw, "text"] = text
          piano_dataset1.at[raw, "level"] = 'elementary'
          raw+=1
    return piano_dataset1


elementary_path = 'C:/Users/Майя/PycharmProjects/Master2021/Samples/Elementrary'
#create_elementary_set
piano_dataset1=create_elementary_set(elementary_path)
print("Elementary set")
print(piano_dataset1)
print()

def create_advanced_set(advanced_path):
    raw = 0
    column = ['name','text', 'level']
    piano_dataset4 = pd.DataFrame(columns=column)

    for folder in os.walk(advanced_path):
      for file in folder[2]:
        read_path = folder[0] + '/' + file
        with open(read_path, 'r') as f:
          text = f.read()
          file_name = file[:-4]
          piano_dataset4.at[raw, "name"] = file_name
          piano_dataset4.at[raw, "text"] = text
          piano_dataset4.at[raw, "level"] = 'advanced'
          raw+=1
    return piano_dataset4


advanced_path='C:/Users/Майя/PycharmProjects/Master2021/Samples/Advanced'
piano_dataset4=create_advanced_set(advanced_path)
print("Advanced set")
print(piano_dataset4)
print()

def create_intermediate_set(intermediate_path):
    raw = 0
    column = ['name','text', 'level']
    piano_dataset2 = pd.DataFrame(columns=column)

    for folder in os.walk(intermediate_path):
      for file in folder[2]:
        read_path = folder[0] + '/' + file
        with open(read_path, 'r') as f:
          text = f.read()
          file_name = file[:-4]
          piano_dataset2.at[raw, "name"] = file_name
          piano_dataset2.at[raw, "text"] = text
          piano_dataset2.at[raw, "level"] = 'intermediate'
          raw+=1
    return piano_dataset2

intermediate_path='C:/Users/Майя/PycharmProjects/Master2021/Samples/Intermediate'
piano_dataset2=create_intermediate_set(intermediate_path)
print("Intermediate set")
print(piano_dataset2)
print()

def create_late_intermediate_set(late_intermediate_path):
    raw = 0
    column = ['name','text', 'level']
    piano_dataset3 = pd.DataFrame(columns=column)

    for folder in os.walk(late_intermediate_path):
      for file in folder[2]:
        read_path = folder[0] + '/' + file
        with open(read_path, 'r') as f:
          text = f.read()
          file_name = file[:-4]
          piano_dataset3.at[raw, "name"] = file_name
          piano_dataset3.at[raw, "text"] = text
          piano_dataset3.at[raw, "level"] = 'late intermediate'
          raw+=1
    return piano_dataset3

intermediate_path='C:/Users/Майя/PycharmProjects/Master2021/Samples/Late Intermediate'
piano_dataset3=create_late_intermediate_set(intermediate_path)
print("Late intermediate set")
print(piano_dataset3)
print()

piano_dataset=pd.concat([piano_dataset1, piano_dataset2, piano_dataset3, piano_dataset4])
shuffled_df = shuffle(piano_dataset)
print("Piano dataset")
print(shuffled_df)
print()
print()
shuffled_df.to_csv("C:/Users/Майя/PycharmProjects/Master2021/piano_dataset.csv")

from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/Майя/PycharmProjects/Master2021/piano_dataset.csv')
y = data.level
X = data.drop('level', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)
print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
print()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tf_idf=TfidfVectorizer(ngram_range=(2,3))

def bow(vectorizer, train, test):
    train_bow=vectorizer.fit_transform(train)
    test_bow=vectorizer.transform(test)
    return train_bow, test_bow

X_train_bow, X_test_bow=bow(vectorizer_tf_idf, list(X_train['text']), list(X_test['text']))
print("Размер матрицы данных обучающей выборки")
X_train_bow=X_train_bow
print(X_train_bow.shape)
print("Размер матрицы данных тестовой выборки")
print(X_test_bow.shape)
print()
print()

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn import metrics

nb = BernoulliNB()
nb.fit(X_train_bow, y_train)

prediction = nb.predict(X_test_bow)
print("Классификация методом Наивного Байеса")
count = 0
for i in X_test['name']:
  line = i+': '+prediction[count]
  print(line)
  count+=1

print()
print("Оценка точности классификации")
print('Naive Bayes')
accuracy_NB=round(accuracy_score(nb.predict(X_test_bow), y_test),2)
print('accuracy_score: ', accuracy_NB)
f1_NB=round(f1_score(y_test, nb.predict(X_test_bow), average='weighted'),2)
#print('f1_score: ', f1_NB)
#target_names=['advanced', 'elementary', 'intermediate', 'late intermediate']
report_NB=classification_report(y_test, nb.predict(X_test_bow), zero_division=0)
print(report_NB)
res_NB='Naive Bayes'+'\n'+'accuracy_score: '+str(accuracy_NB)+'\n'+'classification report: '+ report_NB
print()
print()

clr = LogisticRegression()
clr.fit(X_train_bow, y_train)

prediction = clr.predict(X_test_bow)
print("Классификация методом логистической регрессии")
count = 0
for i in X_test['name']:
  line = i +': '+ prediction[count]
  print(line)
  count+=1

print()
print("Оценка точности классификации")
from sklearn.metrics import classification_report
print('Logistic regression')
accuracy_LR=round(accuracy_score(clr.predict(X_test_bow), y_test),2)
print('accuracy_score: ', accuracy_LR)
f1_LR=round(f1_score(y_test, clr.predict(X_test_bow), average='weighted'),2)
#print('f1_score: ', f1_LR)
report_LR=classification_report(y_test, clr.predict(X_test_bow), zero_division=0)
print(report_LR)
res_LR='Logistic Regression'+'\n'+'accuracy_score: '+str(accuracy_LR)+'\n'+'classification report: '+ report_LR
print()
print()


clf_svc=LinearSVC()
clf_svc.fit(X_train_bow, y_train)

prediction = clf_svc.predict(X_test_bow)
print("Классификация методом опорных векторов")
count = 0
for i in X_test['name']:
    line = i +': '+ prediction[count]
    print(line)
    count+=1

print()
print("Оценка точности классификации")
print('SVM')
accuracy_SVM=round(accuracy_score(clf_svc.predict(X_test_bow), y_test), 2)
print('accuracy_score: ', accuracy_SVM)
f1_SVM=round(f1_score(y_test, clf_svc.predict(X_test_bow), average='weighted'),2)
#print('f1_score: ', f1_SVM)
report_SVM=classification_report(y_test, clf_svc.predict(X_test_bow), zero_division=0)
print(report_SVM)
res_SVM='Support Vector Machine'+'\n'+'accuracy_score: '+str(accuracy_SVM)+'\n'+'classification report: '+ report_SVM
print()
print()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)  #объект класса KNeighborsClassifier, k=3
knn.fit(X_train_bow, y_train)
prediction = knn.predict(X_test_bow)
print("Классификация методом ближайших соседей")
count = 0
with open('C:/Users/Майя/PycharmProjects/Master2021/result_knn.txt', 'w', encoding='utf-8') as result_knn:
  for i in X_test['name']:
    line = i +': '+ prediction[count]
    print(line)
    result_knn.write(line+'\n')
    count+=1

print()
print("Оценка точности классификации")
from sklearn.metrics import classification_report
print('K-Nearest Neighbors')
accuracy_KNN=round(accuracy_score(knn.predict(X_test_bow), y_test),2)
print('accuracy_score: ', accuracy_KNN)
f1_KNN=round(f1_score(y_test, knn.predict(X_test_bow), average='weighted'),2)
#print('f1_score: ', f1_KNN)
report_KNN=classification_report(y_test, knn.predict(X_test_bow), zero_division=0)
print(report_KNN)
res_KNN='K-Nearest Neighbors'+'\n'+'accuracy_score: '+str(accuracy_KNN)+'\n'+'classification report: '+ report_KNN
print()
print()

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train_bow, y_train)
prediction = dtc.predict(X_test_bow)
print("Классификация методом деревьев решений")
count = 0
for i in X_test['name']:
  line = i+': '+prediction[count]
  print(line)
  count+=1

print()
print("Оценка точности классификации")
from sklearn.metrics import classification_report
print('DecisionTreeClassifier')
accuracy_DTC=round(accuracy_score(dtc.predict(X_test_bow), y_test),2)
print('accuracy_score: ', accuracy_DTC)
f1_DTC=round(f1_score(y_test, dtc.predict(X_test_bow), average='weighted'),2)
#print('f1_score: ', f1_DTC)
report_DTC=classification_report(y_test, dtc.predict(X_test_bow), zero_division=0)
print(report_DTC)
res_DTC='Decision Tree Classifier'+'\n'+'accuracy_score: '+str(accuracy_DTC)+'\n'+'classification report: '+ report_DTC
print()
print()




