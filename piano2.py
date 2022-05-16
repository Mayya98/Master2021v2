import os
from os.path import isfile, join
import unicodedata
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
#import pymorphy2
from nltk.collocations import *
import pandas as pd
import numpy as np
import itertools

def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)

def reading_raws(path, path_for_writing):
  files_in_dir = [f for f in os.listdir(path) if isfile(join(path,f))]
  with open(path_for_writing, 'w', encoding='utf-8') as handle1_2w:
    for file in files_in_dir:
      filename=file.replace(file, file[:-4])
      P_files = filename + ".txt"
      with open(path + "/" + P_files, 'r', encoding='utf-8') as handle1_1:
        p_str=handle1_1.read()+ '\n\n'
        handle1_2w.write(p_str)

path='C:/Users/Майя/PycharmProjects/Master2021v2/Piano'
path_for_writing='C:/Users/Майя/PycharmProjects/Master2021v2/p_write.txt'

reading_raws(path, path_for_writing)


with open(path_for_writing, 'r', encoding='utf-8') as handle1_2r:
    desc1_2 = handle1_2r.read()
    corpus = desc1_2.split('\n\n')
    #print(len(corpus))
    corpus = [i for i in corpus if i]
    #print(len(corpus))

def merging(path1):
  with open(path1, 'r', encoding='utf-8') as handle1_2r:
    desc1_2=handle1_2r.read()
    line = re.sub(' in A dur | in A major | A major | A major', ' inA ', desc1_2)
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
    corpus=line.split('\n\n')
    corpus=[a for a in corpus if a]
    return corpus

corpus=merging('C:/Users/Майя/PycharmProjects/Master2021v2/p_write.txt')
#print(len(corpus))

with open("C:/Users/Майя/PycharmProjects/Master2021v2/stops.txt", 'r', encoding='utf-8') as handle_stop:
  stopstring=handle_stop.read()
  stoplist=stopstring.split('\n')

def processing(list_of_texts, stops, processed_path):
  for i in range(len(list_of_texts)):
    text=normalize_caseless(list_of_texts[i])
    list_text = nltk.word_tokenize(text)
    filtered_words = [word for word in list_text if word not in stops if word ]
    filtered_words=[x for x in filtered_words if x]
    joinedstr=" ".join(filtered_words)
    line1 = re.sub(r'[-\(\)\"#\/@;:”“!?\{\}\=\~|\.\?\+\'\[\]``’‘, (\d{1,2}:\d{1,2})]' , ' ', joinedstr)
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

    with open(processed_path, 'w', encoding='utf-8') as handle_pre_w:
      handle_pre_w.write(line1+"\n\n")

list_of_texts=corpus
stops=stoplist
processed_path='C:/Users/Майя/PycharmProjects/Master2021v2/all_write.txt'

"""
with open(processed_path, 'r', encoding='utf-8') as handle_pre_r:
    str_processed_piano=handle_pre_r.read()
    pianomusic=str_processed_piano.split("\n\n")
    pianomusic=[p for p in pianomusic if p]
    print(len(pianomusic))"""


def saving_processed(path, path_for_processed1):
  with open(path, 'r', encoding='utf-8') as handle_pre_r:
    str_processed_piano = handle_pre_r.read()

    pianomusic = str_processed_piano.split('\n\n')
    pianomusic = [i for i in pianomusic if i]

    files_in_dir1 = [f for f in os.listdir(path) if isfile(join(path, f))]

    for file1 in range(len(files_in_dir1)):
      filename1 = files_in_dir1[file1].replace(files_in_dir1[file1], files_in_dir1[file1][:-4])
      Pro_files = filename1 + ".txt"
      with open(path_for_processed1 + "/" + Pro_files, 'w', encoding='utf-8') as  handle2_1w:
        for x in range(len(pianomusic)):
          if x == file1:
            handle2_1w.write(pianomusic[x])

path='C:/Users/Майя/PycharmProjects/Master2021v2/Piano'
path_for_processed1='C:/Users/Майя/PycharmProjects/Master2021v2/Processed Pieces'


def words(path_for_processed0):
  with open(path_for_processed0, 'r', encoding='utf-8') as handle_pre_r:
    str_processed_piano = handle_pre_r.read()
    processed_piano = nltk.word_tokenize(str_processed_piano)
    processed_piano = [i for i in processed_piano if i not in stoplist if i]
    return processed_piano

processed_piano=words('C:/Users/Майя/PycharmProjects/Master2021v2/all_write.txt')
print("Список обработанных слов")
print(processed_piano)
print()


def descr(path_for_processed0):
  with open(path_for_processed0, 'r', encoding='utf-8') as handle_pre_r:
    str_processed_piano = handle_pre_r.read()
    listofd = str_processed_piano.split('\n\n')
    listofd = [i for i in listofd if i]
    return listofd

pianomusic=descr('C:/Users/Майя/PycharmProjects/Master2021v2/all_write.txt')
print("Список описаний")
print(pianomusic)
print()


print("*****Коллокации*****")
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()


def rightTypesBi(ngram):
  acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
  second_type = ('NN', 'NNS', 'NNP', 'NNPS')
  tags = nltk.pos_tag(ngram)
  if tags[0][1] in acceptable_types and tags[1][1] in second_type:
    return True
  else:
    return False

def bigrams(list_of_words):
  finder2 = BigramCollocationFinder.from_words(list_of_words, window_size=2)
  bigramPMITable = pd.DataFrame(list(finder2.score_ngrams(bigram_measures.pmi)), columns=['bigram','PMI'])[:2000].sort_values(by='PMI', ascending=False)

  filtered_PMIbi = bigramPMITable[bigramPMITable.bigram.map(lambda list_of_words: rightTypesBi(list_of_words))]
  PMIbi=filtered_PMIbi.copy()
  PMIbi.PMI=np.round(PMIbi.PMI, 2)
  return PMIbi

PMIbi=bigrams(processed_piano)
print("Биграммы")
print(PMIbi)


def saving_bigrams(bigrams_table, path_bi_txt):
  with open(path_bi_txt, 'w', encoding='utf-8') as bi_PMI_w:
    PMIbi_list = bigrams_table['bigram'].tolist()
    for tpl2 in range(len(PMIbi_list)):
      flatten2 = [str(item) for item in itertools.chain(PMIbi_list[tpl2])]
      PMIbi_str = " ".join(flatten2)
      bi_PMI_w.write(PMIbi_str + '\n')

bigrams_table=PMIbi
path_bi_txt='C:/Users/Майя/PycharmProjects/Master2021v2/all_bigrams.txt'


def rightTypesTri(ngram):
  first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
  third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
  tags = nltk.pos_tag(ngram)
  if tags[0][1] in first_type and tags[2][1] in third_type:
    return True
  else:
    return False

def trigrams(list_of_words):
  finder3 = TrigramCollocationFinder.from_words(list_of_words, window_size=3)
  trigramPMITable = pd.DataFrame(list(finder3.score_ngrams(trigram_measures.pmi)), columns=['trigram','PMI'])[:2000].sort_values(by='PMI', ascending=False)

  filtered_PMItri = trigramPMITable[trigramPMITable.trigram.map(lambda list_of_words: rightTypesTri(list_of_words))]
  PMItri = filtered_PMItri.copy()
  PMItri.PMI=np.round(PMItri.PMI, 2)
  return PMItri

PMItri=trigrams(processed_piano)
print("Триграммы")
print(PMItri)
print()

def saving_trigrams(trigrams_table, path_tri_txt):
  with open(path_tri_txt, 'w', encoding='utf-8') as tri_PMI_w:
    PMItri_list = trigrams_table['trigram'].tolist()
    for tpl3 in range(len(PMItri_list)):
      flatten3 = [str(item) for item in itertools.chain(PMItri_list[tpl3])]
      PMItri_str = " ".join(flatten3)
      tri_PMI_w.write(PMItri_str + '\n')

trigrams_table=PMItri
path_tri_txt='C:/Users/Майя/PycharmProjects/Master2021v2/all_trigrams.txt'

print("Оценка точности метода PMI")
def precision(true_positive, positive):
  return (true_positive / positive) * 100

with open("C:/Users/Майя/PycharmProjects/Master2021v2/Correct ngrams/correct_bigrams.txt", 'r', encoding='utf-8') as handle_correct_bi:
  str_correct_bi=handle_correct_bi.read()
  list_correct_bi=str_correct_bi.split('\n')
  list_correct_bi=[i for i in list_correct_bi if i]
  print("Список правильных биграмм ")
  print(list_correct_bi)
  n_correct_bi=len(list_correct_bi)
  #print(n_correct_bi)
  print()

with open("C:/Users/Майя/PycharmProjects/Master2021v2/all_bigrams.txt", 'r', encoding='utf-8') as handle_all_bi:
  str_all_bi=handle_all_bi.read()
  list_all_bi=str_all_bi.split('\n')
  list_all_bi=[i for i in list_all_bi if i]
  print("Список всех биграмм ")
  print(list_all_bi)
  n_all_bi=len(list_all_bi)
  #print(n_all_bi)
  print()

bigrams_precision=round(precision(n_correct_bi, n_all_bi), 2)
print("Точность по биграммам")
print(bigrams_precision)
print()

with open("C:/Users/Майя/PycharmProjects/Master2021v2/Correct ngrams/correct_trigrams.txt", 'r', encoding='utf-8') as handle_correct_tri:
  str_correct_tri=handle_correct_tri.read()
  list_correct_tri=str_correct_tri.split('\n')
  list_correct_tri=[i for i in list_correct_tri if i]
  print("Список правильных триграмм ")
  print(list_correct_tri)
  n_correct_tri=len(list_correct_tri)
  #print(n_correct_tri)
  print()

with open("C:/Users/Майя/PycharmProjects/Master2021v2/all_trigrams.txt", 'r', encoding='utf-8') as handle_all_tri:
  str_all_tri=handle_all_tri.read()
  list_all_tri=str_all_tri.split('\n')
  list_all_tri=[i for i in list_all_tri if i]
  print("Список всех триграмм ")
  print(list_all_tri)
  n_all_tri=len(list_all_tri)
  #print(n_all_tri)
  print()

trigrams_precision=round(precision(n_correct_tri, n_all_tri), 2)
print("Точность по триграммам")
print(trigrams_precision)