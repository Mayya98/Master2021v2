import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import re
import nltk
#nltk.download('punkt')
import unicodedata

data = pd.read_csv('C:/Users/Майя/PycharmProjects/Master2021/piano_dataset.csv')
y = data.level
X = data.drop('level', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9)


def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)

def merging(str1):
  line = re.sub(' in A dur | in A major | A major | A major', ' inA ', str1)
  line = re.sub(' in a minor | in a moll ', ' inAminor ', line)
  line = re.sub(' in As | in A flat dur | in A flat major ', ' inAs ', line)

  line = re.sub(' in B | in B dur |in B major | B major ', ' inB ', line)
  line = re.sub(' in b | in b minor | in b moll ', ' inBminor ', line)

  line = re.sub(' in C | in C dur | in C major | C major', ' inC ', line)
  line = re.sub(' in c | in cm | cm | in c minor | in c moll ', ' inCminor ', line)
  line = re.sub(' in C# | in Cis dur |in Cis major | Cis major | C# major ', ' inC# ', line)
  line = re.sub(' in c# | in cis moll |in cis minor | cis minor | c# minor ', ' inC#minor ', line)

  line = re.sub(' in D |in D dur| in D major ', ' in D ', line)
  line = re.sub(' in d |in d minor|in d moll', ' inDminor ', line)
  line = re.sub(' in D♭ | in Des dur |in Des major | Des major | D♭ major ', ' inDes ', line)
  line = re.sub(' in d# | in dis moll |in dis minor | dis minor | d# minor ', ' inD#minor ', line)

  line = re.sub(' in E |in E dur | in E major ', ' inE ', line)
  line = re.sub(' in e | in e minor | in e moll ', ' inEminor ', line)
  line = re.sub(' in E♭ | in Ees dur | in Es major | Es major | E♭ major ', ' inEs ', line)
  line = re.sub(' in e♭ | in es moll | in es minor | es minor | e♭ minor ', ' inEsminor ', line)

  line = re.sub(' in F | in F dur| in F major ', ' inF ', line)
  line = re.sub(' in f | in f minor | in f moll ', ' inFminor ', line)
  line = re.sub(' in F# | in Fis dur |in Fis major | Fis major | F# major ', ' inF# ', line)
  line = re.sub(' in f# | in fis moll |in fis minor| fis minor | f# minor ', ' inF#minor ', line)

  line = re.sub(' in G | in Gm | Gm | in G dur | in G major ', ' inG ', line)
  line = re.sub(' in g | in gm | gm | in g minor | in g moll ', ' inGminor ', line)
  line = re.sub(' in G♭ | in Ges dur | in Ges major | Ges major | G♭ major ', ' inGes ', line)
  line = re.sub(' in g# | in gis moll | in gis minor| gis minor | g# minor ', ' inG#minor ', line)

  line = re.sub(' in H | in H dur | in H major ', ' inH ', line)
  line = re.sub(' in h | in h minor | in h moll ', ' inHminor ', line)

  line = re.sub(' in a, | in a. | in a; ', ' inAminor ', line)
  line = re.sub(' in | a ', ' ', line)
  return line


with open("C:/Users/Майя/PycharmProjects/Master2021/stops.txt", 'r', encoding='utf-8') as handle_stop:
  stopstring=handle_stop.read()
  stoplist=stopstring.split('\n')


def processing(str2, stops):
  text=normalize_caseless(str2)
  list_text = nltk.word_tokenize(text)
  filtered_words = [word for word in list_text if word not  in stops if word ]
  filtered_words=[x for x in filtered_words if x]
  joinedstr=" ".join(filtered_words)

  line1 = re.sub(r'[-\(\)\"#\/@;:”“!?\{\}\=\~|\.\?\+\'\[\]`*&’‘,]', ' ', joinedstr)
  line1 = re.sub(' ina ', ' in A major ', line1)
  line1 = re.sub(' inaminor ', ' in a minor ', line1)
  line1 = re.sub(' inas ', ' in A flat major ', line1)

  line1 = re.sub(' inb ', ' in B major ', line1)
  line1 = re.sub(' inbminor ', ' in b minor ', line1)

  line1 = re.sub(' inc ', ' in C major ', line1)
  line1 = re.sub(' incminor ', ' in c minor ', line1)
  line1 = re.sub(' inc# ', ' in Cis major ', line1)
  line1 = re.sub(' inc#minor ', ' in cis minor ', line1)

  line1 = re.sub(' ind ', ' in D major ', line1)
  line1 = re.sub(' indminor ', ' in d minor ', line1)
  line1 = re.sub(' indes ', ' in Des major ', line1)
  line1 = re.sub(' ind#minor ', ' in dis minor ', line1)

  line1 = re.sub(' ine ', ' in E major ', line1)
  line1 = re.sub(' ineminor ', ' in e minor ', line1)
  line1 = re.sub(' ines ', ' in Es major ', line1)
  line1 = re.sub(' inesminor ', ' in es minor ', line1)

  line1 = re.sub(' inf ', ' in F major ', line1)
  line1 = re.sub(' infminor ', ' in f minor ', line1)
  line1 = re.sub(' inf# ', ' in Fis major ', line1)
  line1 = re.sub(' inf#minor ', ' in fis minor ', line1)

  line1 = re.sub(' ing ', ' in G major ', line1)
  line1 = re.sub(' ingminor ', ' in G minor ', line1)
  line1 = re.sub(' inges ', ' in Ges major ', line1)
  line1 = re.sub(' ing#minor ', ' in Gis minor ', line1)

  line1 = re.sub(' inh ', ' in H major ', line1)
  line1 = re.sub(' inhminor ', ' in h minor ', line1)
  line1 = re.sub(' s | – ', ' ', line1)
  line1 = re.sub('  ', ' ', line1)
  return line1

stops=stoplist

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tf_idf=TfidfVectorizer(ngram_range=(2,3))

def bow(vectorizer, train, test):
    train_bow=vectorizer.fit_transform(train)
    test_bow=vectorizer.transform(test)
    return train_bow, test_bow

def determining(str3):
  list1=[str3]
  X_train_bow, X_test_bow=bow(vectorizer_tf_idf, list(X_train['text']), list1)
  knn = KNeighborsClassifier(n_neighbors=3)  # объект класса KNeighborsClassifier, k=3
  knn.fit(X_train_bow, y_train)
  prediction = knn.predict(X_test_bow)
  return prediction[0]

str1='liszt composer known extreme difficulty compositions no 2 rhapsody prime example liszt wrote set 19 hungarian rhapsodies 2 far popular 2 distinct sections re called lassan friska tell difference 2 sections pretty easily lassan section somewhat slow tempo dark dramatic mood friska comes lots trills arpeggios jumps scales brighter mood higher tempo lassan mean easy easier friska section d say ve studied 3 5 years d try friska section bother touching probably good 8 years different people learn different paces ve told able play chopin preludes 6 months playing piano people 2 3 years individualistic thing 2 rhapsody highly highly advance takes competent concert pianist execute right'
res_level=determining(processing(merging(str1), stoplist))
#print(res_level)