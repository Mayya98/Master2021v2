import nltk
import unicodedata
from tkinter import *
from tkinter import messagebox as mb
from test_texts import *
from classification_of_piano_pieces import res_NB, res_LR, res_SVM, res_KNN, res_DTC

def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)

root=Tk()
root.title('Interface for piano lovers')
root.config(bg='ivory2')



with open('C:/Users/Майя/PycharmProjects/Master2021/Results/Elementary.txt', 'r', encoding='utf-8') as handle_elem:
    elem=handle_elem.read()
    elem_li=elem.split('\n')

with open('C:/Users/Майя/PycharmProjects/Master2021/Results/Intermediate.txt', 'r', encoding='utf-8') as handle_inter:
    inter=handle_inter.read()
    inter_li = inter.split('\n')

with open('C:/Users/Майя/PycharmProjects/Master2021/Results/Late Intermediate.txt', 'r', encoding='utf-8') as handle_late:
    late=handle_late.read()
    late_li = late.split('\n')

with open('C:/Users/Майя/PycharmProjects/Master2021/Results/Advanced.txt', 'r', encoding='utf-8') as handle_adv:
    adv=handle_adv.read()
    adv_li = adv.split('\n')

with open('C:/Users/Майя/PycharmProjects/Master2021/titles.txt', 'r', encoding='utf-8') as handle_titles:

    titles=handle_titles.read()
    titles_list=titles.split('\n')

with open('C:/Users/Майя/PycharmProjects/Master2021/Music only.txt', 'r', encoding='utf-8') as handle_music:

    music_terms=handle_music.read().lower()
    music_only=music_terms.split('\n')



greeting=Label(root,text=u'Welcome, dear pianist!',font=('cambria', 16, 'bold'), bg='ivory2')
greeting.pack(in_=root, padx=5, pady=5)

choose=Frame(root, bd=2, padx=2, pady=2, bg='ivory2')
choose.pack(in_=root)
selection_of_repertoire=Button(choose,text=u'Selection of repertoire',font='georgia 14', bg='ivory2')
level_determination=Button(choose,text=u'Determining the level',font='georgia 14', bg='ivory2')
comparing=Button(choose,text=u'Comparing the methods',font='georgia 14', bg='ivory2')

selection_of_repertoire.pack(side="left", in_=choose, padx=5, pady=2)
level_determination.pack(side="left", in_=choose, padx=5, pady=2)
comparing.pack(side="left", in_=choose, padx=5, pady=2)

common_close=Button(root,text=u'CLOSE', height=1,font='georgia 12', bg='ivory2')
common_close.pack(in_=root, side='bottom', padx=2, pady=2)

repertoire=Frame(root,bd=2, padx=2, pady=2, bg='ivory2')
by_level=Button(repertoire,text=u'By level',font='georgia 12', bg='ivory2')
by_composer=Button(repertoire,text=u'By composer',font='georgia 12', bg='ivory2')
by_genre=Button(repertoire,text=u'By genre',font='georgia 12', bg='ivory2')

levels=Frame(root,bd=2, padx=2, pady=2, bg='ivory2')
elementary=Button(levels,text=u'Elementary',font='georgia 11', bg='ivory2')
intermediate=Button(levels,text=u'Intermediate',font='georgia 11', bg='ivory2')
late_intermediate=Button(levels,text=u'Late Intermediate',font='georgia 11', bg='ivory2')
advanced=Button(levels,text=u'Advanced',font='georgia 11', bg='ivory2')

frameforlist=Frame(root,bd=5, width=30, height=50, padx=5, pady=5, bg='ivory2')
list_of_level=Text(frameforlist, height=15, width=28, font='arial 10',wrap=WORD)
close_level_frame=Frame(root,bd=2, width=5, height=3, padx=2, pady=2, bg='ivory2')
close_level_button=Button(close_level_frame,text=u'Close list',font='georgia 12', bg='ivory2')
close_level2=Button(root,text=u'Close selection by level', width=25, font='georgia 12', bg='ivory2')
close_repertoire=Button(root, text=u'Close selection of repertoire', width=25, font='georgia 13', bg='ivory2')

def repertoire_button(event):
    repertoire.pack()
    by_level.pack(side="left", in_=repertoire, padx=2, pady=2)
    by_composer.pack(side="left", in_=repertoire, padx=2, pady=2)
    by_genre.pack(side="left", in_=repertoire, padx=2, pady=2)
    close_repertoire.pack(in_=root, side='bottom')
selection_of_repertoire.bind('<Button-1>', repertoire_button)

def level_button(event):
    levels.pack()
    elementary.pack(side="left", in_=levels, padx=2, pady=2)
    intermediate.pack(side="left", in_=levels, padx=2, pady=2)
    late_intermediate.pack(side="left", in_=levels, padx=2, pady=2)
    advanced.pack(side="left", in_=levels, padx=2, pady=2)
    close_level2.pack(in_=root, side='bottom')
by_level.bind('<Button-1>', level_button)

def elementary_list(event):
    frameforlist.pack()
    list_of_level.insert(1.0, elem)
    list_of_level.pack(in_=frameforlist, padx=2, pady=2)
    close_level_frame.pack()
    close_level_button.pack(in_=close_level_frame, padx=2, pady=2)
elementary.bind('<Button-1>', elementary_list)

def intermediate_list(event):
    frameforlist.pack()
    list_of_level.insert(1.0, inter)
    list_of_level.pack(in_=frameforlist, padx=2, pady=2)
    close_level_frame.pack()
    close_level_button.pack(in_=close_level_frame, padx=2, pady=2)
intermediate.bind('<Button-1>', intermediate_list)

def late_intermediate_list(event):
    frameforlist.pack()
    list_of_level.insert(1.0, late)
    list_of_level.pack(in_=frameforlist, padx=2, pady=2)
    close_level_frame.pack()
    close_level_button.pack(in_=close_level_frame, padx=2, pady=2)
late_intermediate.bind('<Button-1>', late_intermediate_list)

def advanced_list(event):
    frameforlist.pack()
    list_of_level.insert(1.0, adv)
    list_of_level.pack(in_=frameforlist, padx=2, pady=2)
    close_level_frame.pack()
    close_level_button.pack(in_=close_level_frame, padx=2, pady=2)
advanced.bind('<Button-1>', advanced_list)

def close_levels(event):
    list_of_level.delete(1.0, END)
    list_of_level.pack_forget()
    frameforlist.pack_forget()
    close_level_button.pack_forget()
    close_level_frame.pack_forget()
close_level_button.bind('<Button-1>', close_levels)

def close_levels2(event):
    elementary.pack_forget()
    intermediate.pack_forget()
    late_intermediate.pack_forget()
    advanced.pack_forget()
    levels.pack_forget()
    close_level2.pack_forget()
close_level2.bind('<Button-1>', close_levels2)


composers=Frame(root,bd=2, padx=2, pady=2, bg='ivory2')
composer_var=StringVar()
entry_composer=Entry(composers, textvariable=composer_var, width=30, font='arial 10', bg='ivory2')
search_composer=Button(composers,bd=2, width=8, height=1, padx=2, pady=2, bg='ivory2', font='georgia 11', text=u'Search')
close_composers=Button(root,text=u'Close selection by composer', width=30, height=2, font='georgia 12', bg='ivory2')

listofcomposers=Frame(root,bd=5, width=50, height=25, padx=5, pady=5, bg='ivory2')
pieces_by_composer=Text(listofcomposers, height=24, width=50, font='arial 10',wrap=WORD)
clear_entry=Button(composers,bd=2, width=8, height=1, padx=2, pady=2, bg='ivory2', font='georgia 11',text=u'Clear entry')

def composer_button(event):
    composers.pack()
    entry_composer.pack(in_=composers, padx=2, pady=2)
    search_composer.pack(in_=composers, side='left', padx=2, pady=2)
    clear_entry.pack(in_=composers, side='left', padx=2, pady=2)
    close_composers.pack(in_=root, side='bottom')
by_composer.bind('<Button-1>', composer_button)

def search_button(event):
    c=0
    for i in range(len(titles_list)):
        #tokenized_list=nltk.word_tokenize(titles_list[i])
        if normalize_caseless(composer_var.get()) in normalize_caseless(titles_list[i]):
            listofcomposers.pack()
            pieces_by_composer.insert(1.0, titles_list[i]+'\n')
            pieces_by_composer.pack(in_=listofcomposers)
            c+=1
    if c==0:
        info_C = mb.showinfo(title="Info", message="Processing... Choose another composer")
search_composer.bind('<Button-1>', search_button)

def clear_button(event):
    entry_composer.delete(0, END)
    pieces_by_composer.delete(1.0, END)
    pieces_by_composer.pack_forget()
clear_entry.bind('<Button-1>', clear_button)

def close_composers_button(event):
    pieces_by_composer.delete(1.0, END)
    clear_entry.pack_forget()
    pieces_by_composer.pack_forget()
    listofcomposers.pack_forget()
    search_composer.pack_forget()
    entry_composer.pack_forget()
    composers.pack_forget()
    close_composers.pack_forget()
close_composers.bind('<Button-1>', close_composers_button)


genres=Frame(root,bd=2, padx=2, pady=2, bg='ivory2')
genre_var=StringVar()
entry_genre=Entry(genres, textvariable=genre_var, width=30, bg='ivory2')
search_genre=Button(genres,bd=2, width=8, height=1, padx=2, pady=2, bg='ivory2', font='georgia 11', text=u'Search')
close_genres=Button(root,text=u'Close selection by genre', width=30, height=2, font='georgia 12', bg='ivory2')

listofgenres=Frame(root,bd=5, width=40, height=25, padx=5, pady=5, bg='ivory2')
pieces_by_genre=Text(listofgenres, height=24, width=40, font='arial 10',wrap=WORD)
clear_genres_entry=Button(genres,bd=2, width=8, height=1, padx=2, pady=2, bg='ivory2', font='georgia 11',text=u'Clear entry')

def genre_button(event):
    genres.pack()
    entry_genre.pack(in_=genres, padx=2, pady=2)
    search_genre.pack(in_=genres, side='left', padx=2, pady=2)
    clear_genres_entry.pack(in_=genres, side='left', padx=2, pady=2)
    close_genres.pack(in_=root, side='bottom')
by_genre.bind('<Button-1>', genre_button)

def search_genre_button(event):
    c=0
    for i in range(len(titles_list)):
        if normalize_caseless(genre_var.get()) in normalize_caseless(titles_list[i]):
            listofgenres.pack()
            pieces_by_genre.insert(1.0, titles_list[i]+'\n')
            pieces_by_genre.pack(in_=listofgenres)
            c+=1
    if c==0:
        info_G = mb.showinfo(title="Info", message="Processing... Choose another genre")
search_genre.bind('<Button-1>', search_genre_button)

def clear_genre_button(event):
    entry_genre.delete(0, END)
    pieces_by_genre.delete(1.0, END)
    pieces_by_genre.pack_forget()
clear_genres_entry.bind('<Button-1>', clear_genre_button)

def close_genres_button(event):
    pieces_by_genre.delete(1.0, END)
    clear_genres_entry.pack_forget()
    pieces_by_genre.pack_forget()
    listofgenres.pack_forget()
    search_genre.pack_forget()
    entry_genre.pack_forget()
    genres.pack_forget()
    close_genres.pack_forget()
close_genres.bind('<Button-1>', close_genres_button)

def close_repertoire_button(event):
    by_level.pack_forget()
    by_composer.pack_forget()
    by_genre.pack_forget()
    repertoire.pack_forget()
    close_repertoire.pack_forget()
close_repertoire.bind('<Button-1>', close_repertoire_button)


classi_methods1=Frame(root,bd=3, width=700, height=250, padx=4, pady=4, bg='ivory2')
classi_methods2=Frame(root,bd=3, width=700, height=250, padx=4, pady=4, bg='ivory2')
#classi_methods3=Frame(root,bd=3, width=700, height=250, padx=2, pady=2, bg='ivory2')
nb=Text(classi_methods1, bd=3, height=15, width=55, font='arial 10',wrap=WORD)
lr=Text(classi_methods1, bd=3, height=15, width=55, font='arial 10',wrap=WORD)
svm=Text(classi_methods1, bd=3, height=15, width=55, font='arial 10',wrap=WORD)
knn=Text(classi_methods2, bd=3, height=15, width=55, font='arial 10',wrap=WORD)
dtc=Text(classi_methods2, bd=3, height=15, width=55, font='arial 10',wrap=WORD)
close_comparing=Button(classi_methods2,text=u'Close comparing the methods', width=30, height=2, font='georgia 12', bg='ivory2')

def compare_methods(event):
    classi_methods1.pack()
    nb.insert(1.0, res_NB)
    nb.pack(in_=classi_methods1, padx=3, pady=3, side='left')
    lr.insert(1.0, res_LR)
    lr.pack(in_=classi_methods1, padx=3, pady=3, side='left')
    svm.insert(1.0, res_SVM)
    svm.pack(in_=classi_methods1, padx=3, pady=3, side='left')
    classi_methods2.pack()
    knn.insert(1.0, res_KNN)
    knn.pack(in_=classi_methods2, padx=3, pady=3, side='left')
    #classi_methods3.pack()
    dtc.insert(1.0, res_DTC)
    dtc.pack(in_=classi_methods2, padx=3, pady=3, side='left')
    close_comparing.pack(in_=classi_methods2, padx=3, pady=3, side='left')
comparing.bind('<Button-1>', compare_methods)


def close_comparing_button(event):
    #dtc.delete(1.0, res_DTC)
    dtc.pack_forget()
    #knn.delete(1.0, res_KNN)
    knn.pack_forget()
    #classi_methods3.pack_forget()
    #svm.delete(1.0, res_SVM)
    svm.pack_forget()
    classi_methods2.pack_forget()
    #lr.delete(1.0, res_LR)
    lr.pack_forget()
    #nb.delete(1.0, res_NB)
    nb.pack_forget()
    classi_methods1.pack_forget()
    close_comparing.pack_forget()
close_comparing.bind('<Button-1>', close_comparing_button)


determination=Frame(root,bd=2, height=1000, width=1000, padx=2, pady=2, bg='ivory2')
enter_title_label=Label(determination, bd=2, width=30, padx=5, pady=5, bg='ivory2', text=u'Enter a piece of music', font='georgia 12')
enter_title_var=StringVar()
entry_piece=Entry(determination, textvariable=enter_title_var, width=30, bg='ivory2', font='arial 10')
ok=Button(determination,bd=2, width=3, height=1, bg='ivory2', font='georgia 11', text=u'Ok')
level_text1=Text(determination, height=1, width=15, padx=5, pady=5, bg='ivory2', font='arial 11')


enter_desc_label=Label(determination, bd=2, padx=2, pady=2, bg='ivory2', text=u'Enter a description', font='georgia 11')
desc_var=StringVar()
enter_desc_text=Text(determination, height=17, width=40, font='arial 10', wrap=WORD)
ok_full=Button(determination,bd=2, width=3, height=1, padx=5, pady=5, bg='ivory2', font='georgia 11', text=u'OK')
level_text2=Text(determination, height=1, width=15, bg='ivory2', font='arial 11', wrap=WORD)
clear_piece=Button(determination,bd=2, width=3, height=1, padx=5, pady=5, bg='ivory2', font='georgia 11', text=u'Clear')
close_determination=Button(root,text=u'Close level determination', width=30, height=1, padx=3, pady=2, font='georgia 11', bg='ivory2')

def open_testing(event):
    determination.pack(in_=root)
    enter_title_label.pack(in_=determination)
    entry_piece.pack(in_=determination)
    ok.pack(in_=determination)
    close_determination.pack(in_=root, side='bottom')
    clear_piece.pack(in_=determination, side='bottom')
level_determination.bind('<Button-1>', open_testing)

def determine(event):
    entered_piece=enter_title_var.get()
    if entered_piece in titles_list and entered_piece in elem_li:
        level_text1.insert(1.0, 'elementary')
        level_text1.pack(in_=determination)
    if entered_piece in titles_list and entered_piece  in inter_li:
        level_text1.insert(1.0, 'intermediate')
        level_text1.pack(in_=determination)
    if entered_piece in titles_list and entered_piece in late_li:
        level_text1.insert(1.0, 'late intermediate')
        level_text1.pack(in_=determination)
    if entered_piece in titles_list and entered_piece in adv_li:
        level_text1.insert(1.0, 'advanced')
        level_text1.pack(in_=determination)
    if entered_piece not in titles_list:
        enter_desc_label.pack(in_=determination)
        enter_desc_text.pack(in_=determination)
        level_text1.pack_forget()
        ok_full.pack(in_=determination)
ok.bind('<Button-1>', determine)


def big_ok(event):
    k=0
    with open('C:/Users/Майя/PycharmProjects/Master2021v2/InterfaceResults/UsersKNN.txt', 'a', encoding='utf-8') as interf_knn:
        entered_piece=enter_title_var.get()
        entered_desc=enter_desc_text.get('1.0', END)
        #print(entered_desc)
        for m in range(len(music_only)):
            if music_only[m] in entered_desc:
                print(music_only[m])
                k+=1
        if k==0:
            info_M = mb.showinfo(title="Info", message="Entered description doesn't correspond to given subject area. Paste, please another description...")
        else:

            determined_level = determining(processing(merging(str(entered_desc)), stoplist))
            level_text2.insert(1.0, determined_level)
            level_text2.pack(in_=determination)
            knn_level = level_text2.get('1.0', END)
            interf_knn.write(entered_piece + ' : ' + knn_level)
ok_full.bind('<Button-1>', big_ok)

def clear_pieces_button(event):
    level_text2.delete(1.0, END)
    level_text1.delete(1.0, END)
    entry_piece.delete(0, END)
    enter_desc_text.delete(1.0, END)
clear_piece.bind('<Button-1>', clear_pieces_button)

def close_determination_button(event):
    clear_piece.pack_forget()
    level_text2.pack_forget()
    ok_full.pack_forget()
    enter_desc_text.pack_forget()
    enter_desc_label.pack_forget()
    level_text1.pack_forget()
    ok.pack_forget()
    entry_piece.pack_forget()
    enter_title_label.pack_forget()
    close_determination.pack_forget()
close_determination.bind('<Button-1>', close_determination_button)

def quit(event):
    root.destroy()

common_close.bind('<Button-1>', quit)
root.mainloop()
