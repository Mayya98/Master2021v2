import os
from os.path import isfile, join


def descr(path_for_processed0):
  with open(path_for_processed0, 'r', encoding='utf-8') as handle_pre_r:
    str_processed_piano = handle_pre_r.read()
    listofd = str_processed_piano.split('\n\n')
    listofd = [i for i in listofd if i]
    return listofd

pianomusic=descr('C:/Users/Майя/PycharmProjects/Master2021v2/all_write.txt')
print("Processed descriptions")
print(len(pianomusic))
print()

def levels(path_with_levels):
  with open(path_with_levels, 'r', encoding='utf-8') as handle_levels:
    str_levels=handle_levels.read()
    list_levels=str_levels.split('\n')
    list_levels=[i for i in list_levels if i]
    return list_levels

print("Elementary")
elem_list=levels('C:/Users/Майя/PycharmProjects/Master2021v2/Complexity/Elementary.txt')
elem_list=list(set(elem_list))
print(elem_list)
print()

print("Intermediate")
inter_list=levels('C:/Users/Майя/PycharmProjects/Master2021v2/Complexity/Intermediate.txt')
inter_list=list(set(inter_list))
print(inter_list)
print()

print("Late Intermediate")
late_list=levels('C:/Users/Майя/PycharmProjects/Master2021v2/Complexity/Late Intermediate.txt')
late_list=list(set(late_list))
print(late_list)
print()

print("Advanced")
adv_list=levels('C:/Users/Майя/PycharmProjects/Master2021v2/Complexity/Advanced.txt')
adv_list=list(set(adv_list))
print(adv_list)
print()

def save_titles(path_for_processed1, path_with_titles):
  files_in_dir2 = [f for f in os.listdir(path_for_processed1) if isfile(join(path_for_processed1,f))]
  with open(path_with_titles, 'w', encoding='utf-8') as handle_r_w:
    for file2 in files_in_dir2:
      filename2=file2.replace(file2, file2[:-4])
      handle_r_w.write(filename2+'\n')

path_for_processed1='C:/Users/Майя/PycharmProjects/Master2021v2/Processed Pieces'
path_with_titles='C:/Users/Майя/PycharmProjects/Master2021v2/titles.txt'

def titles(path_with_titles0):
  handle_r_r = open(path_with_titles0, 'r', encoding='utf-8')
  titles=handle_r_r.read()
  titles_list=titles.split('\n')
  set_titles=set(titles_list)
  titles_list=list(set_titles)
  titles_list=[i for i in titles_list if i]
  titles_list=sorted(titles_list)
  return titles_list

print("Titles list")
titles_list=titles('C:/Users/Майя/PycharmProjects/Master2021v2/titles.txt')
print(len(titles_list))

"""for p in range(len(pianomusic)):
  for t in range(len(titles_list)):
    if t == p:
      print(t, titles_list[t], p, pianomusic[p])"""

print()
print()

def complexity(list1, list2, level, path_level, path_desc):
  with open (path_level, 'w', encoding='utf-8') as handle_level:
    with open (path_desc, 'w', encoding='utf-8') as handle_desc:
      for x in range(len(list1)):
        for y in range(len(list2)):
          c=0
          for l in range(len(level)):
            if level[l] in list1[x] and x==y:
              print(list2[y], level[l])
              c += 1
          if c!=0:
            handle_level.write(list2[y]+'\n')
            handle_desc.write(list1[x]+'\n')

complexity(pianomusic, titles_list, elem_list, 'C:/Users/Майя/PycharmProjects/Master2021v2/Results/Elementary.txt', 'C:/Users/Майя/PycharmProjects/Master2021v2/descriptions.txt')

def res_levels(res_path):
  with open(res_path, 'r', encoding='utf-8') as handle_res:
    res=handle_res.read()
    list_res=res.split('\n')
    list_res=[i for i in list_res if i]
    return list_res

def de_levels(desc_path):
  with open(desc_path, 'r', encoding='utf-8') as handle_desc:
      desc=handle_desc.read()
      list_desc=desc.split('\n')
      list_desc=[i for i in list_desc if i]
      return list_desc

elementary=res_levels('C:/Users/Майя/PycharmProjects/Master2021v2/Results/Elementary.txt')
#print(elementary)
print()
desc_of_elem=de_levels('C:/Users/Майя/PycharmProjects/Master2021v2/descriptions.txt')
#print(desc_of_elem)

def save_desc(list1, list2, path_for_save):
  for i in range(len(list1)):
    with open(path_for_save + '/' + list1[i] + '.txt', 'w', encoding='utf-8') as h_samp:
      for j in range(len(list2)):
        if j==i:
          h_samp.write(list2[j])

save_desc(elementary, desc_of_elem, 'C:/Users/Майя/PycharmProjects/Master2021v2/Samples/Elementrary')

titles_list=[t for t in titles_list if t not in elementary]
#print(len(titles_list))
print()
pianomusic=[p for p in pianomusic if p not in desc_of_elem]
#print(len(pianomusic))

complexity(pianomusic, titles_list, adv_list, 'C:/Users/Майя/PycharmProjects/Master2021v2/Results/Advanced.txt', 'C:/Users/Майя/PycharmProjects/Master2021/descriptions.txt')

advanced=res_levels('C:/Users/Майя/PycharmProjects/Master2021v2/Results/Advanced.txt')
#print(len(advanced))
print()
desc_of_adv=de_levels('C:/Users/Майя/PycharmProjects/Master2021v2/descriptions.txt')
#print(len(desc_of_adv))

save_desc(advanced, desc_of_adv, 'C:/Users/Майя/PycharmProjects/Master2021v2/Samples/Advanced')

titles_list=[t for t in titles_list if t not in advanced]
#print(titles_list)
print()
pianomusic=[p for p in pianomusic if p not in desc_of_adv]
#print(pianomusic)

complexity(pianomusic, titles_list, inter_list, 'C:/Users/Майя/PycharmProjects/Master2021v2/Results/Intermediate.txt', 'C:/Users/Майя/PycharmProjects/Master2021v2/descriptions.txt')
intermediate=res_levels('C:/Users/Майя/PycharmProjects/Master2021v2/Results/Intermediate.txt')
#print(intermediate)
print()
desc_of_inter=de_levels('C:/Users/Майя/PycharmProjects/Master2021v2/descriptions.txt')
#print(len(desc_of_inter))
print()

save_desc(intermediate, desc_of_inter, 'C:/Users/Майя/PycharmProjects/Master2021v2/Samples/Intermediate')

titles_list=[t for t in titles_list if t not in intermediate]
#print(titles_list)
print()
pianomusic=[p for p in pianomusic if p not in desc_of_inter]
#print(pianomusic)

samples_path_late='C:/Users/Майя/PycharmProjects/Master2021v2/Samples/Late Intermediate'
with open('C:/Users/Майя/PycharmProjects/Master2021v2/Results/Late Intermediate.txt', 'w', encoding='utf-8') as handle_late:
  for t in range(len(titles_list)):
    handle_late.write(titles_list[t]+'\n')
    with open(samples_path_late + '/' + titles_list[t] + '.txt', 'w', encoding='utf-8') as h_samp_late:
      for p in range(len(pianomusic)):
        if p==t:
            h_samp_late.write(pianomusic[p])

