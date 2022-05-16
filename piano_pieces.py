#импорты библиотек и модулей
import os
from os.path import isfile, join
import unicodedata
import re
import nltk
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

#print(caseless_equal('Mozart', 'mozart'))
#print()


#список папок
path='C:/Users/Майя/PycharmProjects/Master2021/Piano'

print()
print("*****Piano***** ")
files_in_dir = [f for f in os.listdir(path) if isfile(join(path,f))]
handle1_2w=open("C:/Users/Майя/PycharmProjects/Master2021/p_write.txt", 'w', encoding='utf-8')
print("Файлы с описанием произведений для фортепиано: ")
for file in files_in_dir:
    filename=file.replace(file, file[:-4])
    P_files = filename + ".txt"
    print(P_files)
    handle1_1 = open(path + "/" + P_files, 'r', encoding='utf-8')
    p_str=handle1_1.read()+ '\n\n'
    handle1_2w.write(p_str)
handle1_2w.close()
handle1_1.close()
print()

handle1_2r = open("C:/Users/Майя/PycharmProjects/Collins/p_write.txt", 'r', encoding='utf-8')
print("Список описаний произведений для фортепиано: ")
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
handle1_2r.close()
#print(line+'\n')
corpus=line.split('\n\n')
corpus=[a for a in corpus if a]

print(len(corpus))
print()

handle_pre_w=open("C:/Users/Майя/PycharmProjects/Master2021/all_write.txt", 'w', encoding='utf-8')

print("*****Обработка*****")
print()

handle_stop=open("C:/Users/Майя/PycharmProjects/Master2021/stops.txt", 'r', encoding='utf-8')
stopstring=handle_stop.read()
stoplist=stopstring.split('\n')


for i in range(len(corpus)):
    text=normalize_caseless(corpus[i])
    #print(text)
    list_text = nltk.word_tokenize(text)
    #print(list_text)
    filtered_words = [word for word in list_text if word not  in stoplist if word ]
    filtered_words=[x for x in filtered_words if x]
    joinedstr=" ".join(filtered_words)

    line1 = re.sub(r'[-\(\)\"#\/@;:”“!?\{\}\=\~|\.\?\+\'\[\]`*&’‘,]', ' ', joinedstr)
    #print(line1)
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

    #print(line1+'\n')
    handle_pre_w.write(line1+"\n\n")
handle_pre_w.close()

print()
print()

handle_pre_r=open("C:/Users/Майя/PycharmProjects/Master2021/all_write.txt", 'r', encoding='utf-8')
str_processed_piano=handle_pre_r.read()
processed_piano=nltk.word_tokenize(str_processed_piano)
processed_piano=[i for i in processed_piano if i not  in stoplist if i]
print("Обработанные описания произведений для фортепиано: ")
print(len(processed_piano))
handle_pre_r.close()
print()


#список папок
path_for_processed='C:/Users/Майя/PycharmProjects/Master2021/Processed Pieces'

print("Cписок обработанны[ описания произведений для фортепиано: ")
pianomusic=str_processed_piano.split('\n\n')
pianomusic=list(filter(None, pianomusic))
#print(len(pianomusic))
for i in range(len(pianomusic)):
    print(i, pianomusic[i])

print()
files_in_dir1 = [f for f in os.listdir(path) if isfile(join(path,f))]
files_in_dir1=sorted(files_in_dir1)
print(files_in_dir1)
print("Файлы с описанием произведений для фортепиано: ")
for file1 in range(len(files_in_dir1)):
    filename1=files_in_dir1[file1].replace(files_in_dir1[file1], files_in_dir1[file1][:-4])
    Pro_files = filename1 + ".txt"
    print(Pro_files)
    handle2_1w = open(path_for_processed + "/" + Pro_files, 'w', encoding='utf-8')
    for x in range(len(pianomusic)):
        if x==file1:
            handle2_1w.write(pianomusic[x])

handle2_1w.close()
print()


print()
print()

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
#quadgram_measures = nltk.collocations.QuadgramAssocMeasures()

finder2 = BigramCollocationFinder.from_words(processed_piano, window_size=2)
finder3 =TrigramCollocationFinder.from_words(processed_piano, window_size=3)

print()
print("********Биграммы с применением фильтра POS-tagging********")
def rightTypesBi(ngram):
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

#print("Список биграмм, отсортированных по убыванию pmi")
bigrams = finder2.nbest(bigram_measures.pmi, 1000)
print(len(bigrams))
print()

print("Биграммы и их значения pmi")
for y in bigrams:
    b=y, round(finder2.score_ngram(bigram_measures.pmi, y[0], y[1]),2)
    print(b)

print()
print("Биграммы и их значения pmi в виде таблицы")
bigramPMITable = pd.DataFrame(list(finder2.score_ngrams(bigram_measures.pmi)), columns=['bigram','PMI'])[:2000].sort_values(by='PMI', ascending=False)

filtered_PMIbi = bigramPMITable[bigramPMITable.bigram.map(lambda processed_piano: rightTypesBi(processed_piano))]
PMIbi=filtered_PMIbi.copy()
PMIbi['PMI']=np.round(PMIbi['PMI'], 2)
print(PMIbi)
PMIbi.to_csv("PMIbi.csv", index=False, sep="|")
print()

bi_PMI_w=open("C:/Users/Майя/PycharmProjects/Master2021/all_bigrams.txt", 'w', encoding='utf-8')
PMIbi_list=PMIbi['bigram'].tolist()
print(PMIbi['bigram'])
for tpl2 in range (len(PMIbi_list)):
    #print(PMIbi_list[tpl2])
    flatten2 = [str(item) for item in itertools.chain(PMIbi_list[tpl2])]
    PMIbi_str=" ".join(flatten2)
    #print(PMIbi_str)
    bi_PMI_w.write(PMIbi_str+'\n')
bi_PMI_w.close()
print()

print("********Триграммы с применением фильтра POS-tagging********")
print("Триграммы с их частотой")
def rightTypesTri(ngram):
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
#filter trigrams

print("Список триграмм, отсортированных по убыванию pmi")
trigrams = finder3.nbest(trigram_measures.pmi, 1000)
print(len(trigrams))
print()

print("Триграммы и их значения pmi")
for x in trigrams:
    print(x, round(finder3.score_ngram(trigram_measures.pmi, x[0], x[1], x[2]),2))

print()
print("Триграммы и их значения pmi в виде таблицы")
trigramPMITable = pd.DataFrame(list(finder3.score_ngrams(trigram_measures.pmi)), columns=['trigram','PMI'])[:2000].sort_values(by='PMI', ascending=False)

filtered_PMItri = trigramPMITable[trigramPMITable.trigram.map(lambda processed_piano: rightTypesTri(processed_piano))]
PMItri = filtered_PMItri.copy()
PMItri.PMI=np.round(PMItri.PMI, 2)
print(PMItri)
PMItri.to_csv("PMItri.csv", index=False, sep="|")
print()

tri_PMI_w=open("C:/Users/Майя/PycharmProjects/Master2021/all_trigrams.txt", 'w', encoding='utf-8')
PMItri_list=PMItri['trigram'].tolist()
for tpl3 in range (len(PMItri_list)):
    #print(PMItri_list[tpll3])
    flatten = [str(item) for item in itertools.chain(PMItri_list[tpl3])]
    PMItri_str=" ".join(flatten)
    #print(PMItri_str)
    tri_PMI_w.write(PMItri_str+'\n')
tri_PMI_w.close()
print()


def precision(true_positive, positive):
    return (true_positive/positive)*100

handle_correct_bi=open("C:/Users/Майя/PycharmProjects/Master2021/Correct ngrams/correct_bigrams.txt", 'r', encoding='utf-8')
str_correct_bi=handle_correct_bi.read()
list_correct_bi=str_correct_bi.split('\n')
list_correct_bi=[i for i in list_correct_bi if i]
list_correct_bi=list(set(list_correct_bi))
print("Список правильно выданных биграмм: ")
print(list_correct_bi)
n_correct_bi=len(list_correct_bi)
print()

handle_all_bi=open("C:/Users/Майя/PycharmProjects/Master2021/all_bigrams.txt", 'r', encoding='utf-8')
str_all_bi=handle_all_bi.read()
list_all_bi=str_all_bi.split('\n')
list_all_bi=[i for i in list_all_bi if i]
#print("Список всех выданных биграмм: ")
#print(list_all_bi)
n_all_bi=len(list_all_bi)
#print()
print("Точность по биграммам: ")
bigrams_precision=round(precision(n_correct_bi, n_all_bi), 2)
print(bigrams_precision)
print()


handle_correct_tri=open("C:/Users/Майя/PycharmProjects/Master2021/Correct ngrams/correct_trigrams.txt", 'r', encoding='utf-8')
str_correct_tri=handle_correct_tri.read()
list_correct_tri=str_correct_tri.split('\n')
list_correct_tri=[i for i in list_correct_tri if i]
list_correct_tri=list(set(list_correct_tri))
print("Список правильно выданных биграмм: ")
print(list_correct_tri)
n_correct_tri=len(list_correct_tri)
print()


handle_all_tri=open("C:/Users/Майя/PycharmProjects/Master2021/all_trigrams.txt", 'r', encoding='utf-8')
str_all_tri=handle_all_tri.read()
list_all_tri=str_all_tri.split('\n')
list_all_tri=[i for i in list_all_tri if i]
#print("Список всех выданных биграмм: ")
#print(list_all_tri)
n_all_tri=len(list_all_tri)
#print()
print("Точность по триграммам: ")
trigrams_precision=round(precision(n_correct_tri, n_all_tri), 2)
print(trigrams_precision)
print()
print()


handle_elem_bi=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Bigrams/Elementary.txt", 'r', encoding='utf-8')
str_elem_bi=handle_elem_bi.read()
list_elem_bi=str_elem_bi.split('\n')
list_elem_bi=[i for i in list_elem_bi if i]
print("Список биграмм, соответствующих уровню Elementary: ")
print(list_elem_bi)
print()



handle_inter_bi=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Bigrams/Intermediate.txt", 'r', encoding='utf-8')
str_inter_bi=handle_inter_bi.read()
list_inter_bi=str_inter_bi.split('\n')
list_inter_bi=[i for i in list_inter_bi if i]
print("Список биграмм, соответствующих уровню Intermediate: ")
print(list_inter_bi)
print()


handle_late_bi=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Bigrams/Late Intermediate.txt", 'r', encoding='utf-8')
str_late_bi=handle_late_bi.read()
list_late_bi=str_late_bi.split('\n')
list_late_bi=[i for i in list_late_bi if i]
print("Список биграмм, соответствующих уровню Late Intermediate: ")
print(list_late_bi)
print()

handle_adv_bi=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Bigrams/Advanced.txt", 'r', encoding='utf-8')
str_adv_bi=handle_adv_bi.read()
list_adv_bi=str_adv_bi.split('\n')
list_adv_bi=[i for i in list_adv_bi if i]
print("Список биграмм, соответствующих уровню Advanced: ")
print(list_adv_bi)
print()

print("************************Trigrams***************************")

handle_elem_tri=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Trigrams/Elementary.txt", 'r', encoding='utf-8')
str_elem_tri=handle_elem_tri.read()
list_elem_tri=str_elem_tri.split('\n')
list_elem_tri=[i for i in list_elem_tri if i]
print("Список триграмм, соответствующих уровню Elementary: ")
print(list_elem_tri)
print()

handle_inter_tri=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Trigrams/Intermediate.txt", 'r', encoding='utf-8')
str_inter_tri=handle_inter_tri.read()
list_inter_tri=str_inter_tri.split('\n')
list_inter_tri=[i for i in list_inter_tri if i]
print("Список триграмм, соответствующих уровню Intermediate: ")
print(list_inter_tri)
print()

handle_late_tri=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Trigrams/Late Intermediate.txt", 'r', encoding='utf-8')
str_late_tri=handle_late_tri.read()
list_late_tri=str_late_tri.split('\n')
list_late_tri=[i for i in list_late_tri if i]
print("Список триграмм, соответствующих уровню Late Intermediate: ")
print(list_late_tri)
print()

handle_adv_tri=open("C:/Users/Майя/PycharmProjects/Master2021/Complexity/Trigrams/Advanced.txt", 'r', encoding='utf-8')
str_adv_tri=handle_adv_tri.read()
list_adv_tri=str_adv_tri.split('\n')
list_adv_tri=[i for i in list_adv_tri if i]
print("Список триграмм, соответствующих уровню Advanced: ")
print(list_adv_tri)
print()

print("*****************************************************")

elem_list=list_elem_bi+list_elem_tri
print(elem_list)
print()

inter_list=list_inter_bi+list_inter_tri
print(inter_list)
print()

late_list=list_late_bi+list_late_tri
print(late_list)
print()

adv_list=list_adv_bi+list_adv_tri
print(adv_list)
print()



print()
files_in_dir2 = [f for f in os.listdir(path_for_processed) if isfile(join(path_for_processed,f))]
handle_r_w=open("C:/Users/Майя/PycharmProjects/Master2021/titles.txt", 'w', encoding='utf-8')
#print("Файлы с описанием произведений для фортепиано: ")
for file2 in files_in_dir2:
    filename2=file2.replace(file2, file2[:-4])
    #print(filename2)
    handle_r_w.write(filename2+'\n')
handle_r_w.close()
#print()


print("Список названий")
handle_r_r = open("C:/Users/Майя/PycharmProjects/Master2021/titles.txt", 'r', encoding='utf-8')
titles=handle_r_r.read()
titles_list=titles.split('\n')
set_titles=set(titles_list)
titles_list=list(set_titles)
titles_list=[i for i in titles_list if i]
titles_list=sorted(titles_list)
#print(titles_list)
print()


handle_elem = open("C:/Users/Майя/PycharmProjects/Master2021/Results/Elementary.txt", 'w', encoding='utf-8')
handle_inter= open("C:/Users/Майя/PycharmProjects/Master2021/Results/Intermediate.txt", 'w', encoding='utf-8')
handle_late= open("C:/Users/Майя/PycharmProjects/Master2021/Results/Late Intermediate.txt", 'w', encoding='utf-8')
handle_adv = open("C:/Users/Майя/PycharmProjects/Master2021/Results/Advanced.txt", 'w', encoding='utf-8')
handle_desc = open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'w', encoding='utf-8')



for p in range(len(pianomusic)):
    for t in range(len(titles_list)):
       c = 0
       for e in range(len(elem_list)):
           #print(elem_list[e])
           if elem_list[e] in pianomusic[p] and p==t:
                c += 1
                #h_samp_elem.write(pianomusic[p])
                print(titles_list[t], elem_list[e])

       if c!=0:
            handle_elem.write(titles_list[t]+'\n')
            handle_desc.write(pianomusic[p]+'\n\n')


handle_elem.close()
handle_desc.close()


handle_res_e=open("C:/Users/Майя/PycharmProjects/Master2021/Results/Elementary.txt", 'r', encoding='utf-8')
handle_desc_e=open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'r', encoding='utf-8')

res_e=handle_res_e.read()
list_res_e=res_e.split('\n')
list_res_e=[i for i in list_res_e if i]
print(list_res_e)
desc_e=handle_desc_e.read()
list_desc_e=desc_e.split('\n\n')
list_desc_e=[i for i in list_desc_e if i]
print(list_desc_e)
print()

samples_path_elem='C:/Users/Майя/PycharmProjects/Master2021/Samples/Elementrary'
for el in range(len(list_res_e)):
    h_samp_elem = open(samples_path_elem + '/' + list_res_e[el] + '.txt', 'w', encoding='utf-8')
    for desc_el in range(len(list_desc_e)):
        if desc_el==el:
            h_samp_elem.write(list_desc_e[desc_el])
h_samp_elem.close()



titles_list=[p for p in titles_list if p not in list_res_e]
print(titles_list)
print()

pianomusic=[p for p in pianomusic if p not in list_desc_e]
print(len(pianomusic))
print()
print()

handle_desc = open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'w', encoding='utf-8')
for p in range(len(pianomusic)):
    for t in range(len(titles_list)):
        c=0
        for a in range(len(adv_list)):
            if adv_list[a] in pianomusic[p] and p==t:
                #print(titles_list[t], adv_list[a])
                c+=1

        if c!=0:

            handle_adv.write(titles_list[t]+'\n')
            handle_desc.write(pianomusic[p] + '\n\n')
handle_adv.close()
handle_desc.close()



handle_res_a=open("C:/Users/Майя/PycharmProjects/Master2021/Results/Advanced.txt", 'r', encoding='utf-8')
handle_desc_a=open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'r', encoding='utf-8')

res_a=handle_res_a.read()
list_res_a=res_a.split('\n')
list_res_a=[i for i in list_res_a if i]
print(list_res_a)
desc_a=handle_desc_a.read()
list_desc_a=desc_a.split('\n\n')
list_desc_a=[i for i in list_desc_a if i]
print(list_desc_a)
print()

samples_path_adv='C:/Users/Майя/PycharmProjects/Master2021/Samples/Advanced'
for adv in range(len(list_res_a)):
    h_samp_adv = open(samples_path_adv + '/' + list_res_a[adv] + '.txt', 'w', encoding='utf-8')
    for desc_adv in range(len(list_desc_a)):
        if desc_adv==adv:
            h_samp_adv.write(list_desc_a[desc_adv])
h_samp_adv.close()



titles_list=[p for p in titles_list if p not in list_res_a]
print(titles_list)
print()

pianomusic=[p for p in pianomusic if p not in list_desc_a]
print(len(pianomusic))
handle_res_a.close()
handle_desc_a.close()
print()
print()

handle_desc = open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'w', encoding='utf-8')
for p in range(len(pianomusic)):
    for t in range(len(titles_list)):
       c = 0
       for i in range(len(inter_list)):
            if inter_list[i] in pianomusic[p] and t==p:
                print(titles_list[t], inter_list[i])
                c += 1
       if c!=0:
            handle_inter.write(titles_list[t]+'\n')
            handle_desc.write(pianomusic[p] + '\n\n')
handle_inter.close()
handle_desc.close()

handle_res_i=open("C:/Users/Майя/PycharmProjects/Master2021/Results/Intermediate.txt", 'r', encoding='utf-8')
handle_desc_i=open("C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt", 'r', encoding='utf-8')

res_i=handle_res_i.read()
list_res_i=res_i.split('\n')
list_res_i=[i for i in list_res_i if i]
print(list_res_i)
desc_i=handle_desc_i.read()
list_desc_i=desc_i.split('\n\n')
list_desc_i=[i for i in list_desc_i if i]
print(list_desc_i)
print()

handle_res_i.close()
handle_desc_i.close()

samples_path_inter='C:/Users/Майя/PycharmProjects/Master2021/Samples/Intermediate'
for inter in range(len(list_res_i)):
    h_samp_inter = open(samples_path_inter + '/' + list_res_i[inter] + '.txt', 'w', encoding='utf-8')
    for desc_inter in range(len(list_desc_i)):
        if desc_inter==inter:
            h_samp_inter.write(list_desc_i[desc_inter])
h_samp_inter.close()


titles_list=[p for p in titles_list if p not in list_res_i]
print(titles_list)
print()

pianomusic=[p for p in pianomusic if p not in list_desc_i]
print(pianomusic)
print()
print()


samples_path_late='C:/Users/Майя/PycharmProjects/Master2021/Samples/Late Intermediate'
for t in range(len(titles_list)):
    handle_late.write(titles_list[t]+'\n')
    h_samp_late = open(samples_path_late + '/' + titles_list[t] + '.txt', 'w', encoding='utf-8')
    for p in range(len(pianomusic)):
        if p==t:
            h_samp_late.write(pianomusic[p])
handle_late.close()
h_samp_late.close()
