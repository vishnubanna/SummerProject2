#!/usr/bin/env python3

from __future__ import unicode_literals

from bs4 import BeautifulSoup
from urllib.request import urlopen

import re
from newspaper import Article
import nltk
from nltk.tree import Tree
from nltk.corpus import stopwords
import spacy
from spacy import displacy
import stanza
from spacy_stanza import StanzaLanguage # the stanford nlp librarys
from spacy.util import get_lang_class

import youtube_dl
import speech_recognition as sr

import numpy as np
import math
import os
import sys

nltk.download('punkt')
nltk.download('stopwords')
STOP = stopwords.words('english')

def load_model(model = "en", group = "stanford"):
    if group == "stanford": 
        # lang_cls = get_lang_class("stanza_en")
        stanza.download('en')
        snlp = stanza.Pipeline(lang="en", use_gpu=True)
        nlp = StanzaLanguage(snlp)
    elif group == None:
        nlp = spacy.load(model)
    return nlp

def load_sentence_splitter():
    nlp = spacy.load('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
    return nlp

def format_sentence(sent):
    sent = re.sub(r"[\“\”]", "", sent)
    sent = re.sub(r"([\\][x](.){2})","", sent)
    sent = re.sub(r"[\’]", "'", sent)
    sent = re.sub(r"(\')", "'", sent)
    sent = re.sub(r"[<](.)[>]", " ", sent)
    sent = re.sub(r"[\n]+", " ", sent)
    sent = re.sub(r"[\t]+", " ", sent)
    sent = re.sub(r"[ ]+", " ", sent)
    # sent = re.sub(r"([0-9]+ Comments )(.)+", "", sent)
    return sent

def format_sentence_2(sent):
    sent = re.sub(r"[\“\”]", "", sent)
    sent = re.sub(r"[\’]", "'", sent)
    sent = re.sub(r"(\')", "'", sent)
    sent = re.sub(r"[<](.)[>]", "", sent)
    sent = re.sub(r"[\n]+", "\n", sent)
    sent = re.sub(r"[\t]+", " ", sent)
    sent = re.sub(r"[ ]+", " ", sent)
    # sent = re.sub(r"([0-9]+ Comments )(.)+", "", sent)
    return sent

def token_counts(model, sent):
    sent = model(sent)
    tokens = []
    rem = {"/", " ", ""}
    count = 0 
    for word in sent:
        if word.lemma_ not in STOP and word.pos_ != "PUNCT" and "'" not in word.text and word.text not in rem:
            if word.lemma_ != "-PRON-":
                tokens.append(word.lemma_)
            else:
                tokens.append(word.text)
        else:
            count += 1
    return count, tokens

def tokenize(model, sent):
    sent = model(sent)
    tokens = []
    rem = {"/", " ", ""}
    for word in sent:
        if word.lemma_ not in STOP and word.pos_ != "PUNCT" and "'" not in word.text and word.text not in rem:
            tokens.append(word.lemma_)
    return tokens

def filter_line(line):
    if (len(line) < 0):
        return False
    if ('{' in line or '}' in line or 'var' in line):
        return False
    if (re.search(r"((\.)[a-zA-Z])", line) != None):
        return False
    if ('©' in line):
        return False
    return True

def sort_dict(vals):
    return {key: value for key, value in sorted(vals.items(), key = lambda x: x[1], reverse=True)}

def gen_sets(val_set):
    paradicts = []
    for sentset in val_set:
        glob_set = set()
        for item in sentset:
            sent = set(item.split(" "))
            glob_set.update(sent)
        paradicts.append(glob_set)
    return paradicts

def include(text, keys):
    text = set(text)
    keys = set(keys)

    overlap = text.intersection(keys)
    # print(len(overlap))

    return len(overlap) > 5 

def general_library(url, threshold):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    text = format_sentence_2(text)
    return text

def merge_sets(keys):
    glob_set = set()
    for key in keys:
        glob_set.update(key)
    return glob_set

def general_statistical(url, threshold):
    """
    todo: build the filter network

    ideas:
        - maybe if the title is in the sentence definitly include it
        - maybe try a regression 
        - if not use a neral network with glove ideally it will use word similarity to find related texts
    """
    nlp = load_model(group = None)
    #nlp = load_model()
    sent_splitter = load_sentence_splitter()

    text = urlopen(url)
    site = text.read()
    text.close()
    
    items_all = BeautifulSoup(site, features="lxml")
    items = items_all.find_all('div')
    print(type(items))
    none = {"\n", "", " "}
    paragraphs = []
    raw_sentences = []
    stp_count = []
    for lines in items:
        text = format_sentence(lines.text)
        sum_counts = []
        para_sents = []
        if filter_line(text) and text not in none:
            for sent in sent_splitter(str(text)).sents:
                count, tokens = token_counts(nlp, sent.string.strip().lower())
                if len(tokens) > 0:
                    sum_counts.append(count)
                    para_sents.append(format_sentence(" ".join(tokens)))
            para_sents = set(para_sents)
            if para_sents not in paragraphs:
                paragraphs.append(para_sents)
                stp_count.append(sum(sum_counts))
                raw_sentences.append(text)

    # paragraphs = np.array(gen_sets(paragraphs))
    stp_count = np.array(stp_count)
    raw_sentences = np.array(raw_sentences)
    index = np.where(stp_count > threshold)[0]

    # keys = merge_sets(paragraphs[index])
    # for i, text in enumerate(paragraphs):
    #     print(include(text, keys), stp_count[i])
    # print(keys)

    print(stp_count)
    return re.sub(r"[ ]+"," ","\n\n".join(list(raw_sentences[index])))

def general_statistical_ptag(url, threshold):
    """
    todo: build the filter network
          find a way to automatically switch between paragraphtags and div tags
          if there is paragraph tags in a large div tag, use _ptag other wise use general_statistical

    ideas:
        - maybe if the title is in the sentence definitly include it
        - maybe try a regression 
        - if not use a neral network with glove ideally it will use word similarity to find related texts
    """
    nlp = load_model(group = None)
    #nlp = load_model()
    sent_splitter = load_sentence_splitter()

    text = urlopen(url)
    site = text.read()
    text.close()
    
    items_all = BeautifulSoup(site, features="lxml")
    items = items_all.find_all('p')
    none = {"\n", "", " "}
    paragraphs = []
    raw_sentences = []
    stp_count = []
    for lines in items:
        text = format_sentence(lines.text)
        sum_counts = []
        para_sents = []
        if filter_line(text) and text not in none:
            for sent in sent_splitter(str(text)).sents:
                count, tokens = token_counts(nlp, sent.string.strip().lower())
                if len(tokens) > 0:
                    sum_counts.append(count)
                    para_sents.append(format_sentence(" ".join(tokens)))
            para_sents = set(para_sents)
            if para_sents not in paragraphs:
                paragraphs.append(para_sents)
                stp_count.append(sum(sum_counts))
                raw_sentences.append(text)

    # paragraphs = np.array(gen_sets(paragraphs))
    stp_count = np.array(stp_count)
    raw_sentences = np.array(raw_sentences)
    index = np.where(stp_count > threshold)[0]

    # keys = merge_sets(paragraphs[index])
    # for i, text in enumerate(paragraphs):
    #     print(include(text, keys), stp_count[i])
    # print(keys)

    print(stp_count)
    return re.sub(r"[ ]+"," ","\n\n".join(list(raw_sentences[index])))

def get_youtube(url, threshold):
    """
    TODO: 
        use the google cloud api or microsoft api instead of the chunk method, it is faster and more accurate
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.download([url])
        info = ydl.extract_info(url, download=False)

    filename = re.sub(r"[:]", " -", f"{info['title']}-{info['id']}.wav")
    
    r = sr.Recognizer()
    source = sr.AudioFile(filename)

    with source as s:
        print(s.SAMPLE_RATE)
        length = math.ceil(s.DURATION/threshold)
        print(s.DURATION, length)
        ret = []
        for i in range(length):
            audio = r.record(s, duration = threshold)
            ret.append(audio)

    words = []
    for chunk in ret:
        spoken = r.recognize_google(chunk, show_all=True)
        try:
            words.append(spoken["alternative"][-1]["transcript"])
        except:
            print(spoken)

    os.system(f"rm {filename}")
    # print("\n".join(words))
    # print(words[0])
    return "\n".join(words)

def search_url(url):
    global methods
    url = re.sub(r"(https://)|(http://)", "http://www.", url)
    urlset = set(url.split("."))
    company = list(urlset.intersection(methods))
    return company[0] if len(company) > 0 else "None"

def load_article(url):
    global method_reader
    company = search_url(url)
    if company in method_reader.keys():
        return method_reader[company][0](url, method_reader[company][1])
    else:
        raise RuntimeError("unsupported website")


# wsj doesn't work fix me
methods = {"guardian", "variety", "youtube", "rogerebert", "nytimes", "wsj"}
method_reader = {"guardian": [general_library, None], "variety": [general_library, None], "youtube":[get_youtube, 10], "rogerebert": [general_statistical_ptag, 5], "nytimes": [general_statistical_ptag, 5], "None": [general_statistical, 15]}


def CLI_run():
    inputs = sys.argv[:]

    if len(inputs) > 3 + 1:
        raise RuntimeError("too many inputs -> input1 is the link -> input2 is the function -> input3 is the threshold (none if statistical method is not used)")

    if len(inputs) < 1 + 1:
        raise RuntimeError("too few inputs -> input1 is the link -> input2 is the function -> input3 is the threshold (none if statistical method is not used)")

    url = sys.argv[1]
    print(len(sys.argv))
    if (len(sys.argv) > 2):
        if sys.argv[2] == "general_library":
            text = general_library(url, None)
        elif sys.argv[2] == "general_statistical_ptag":   
            try: 
                text = general_statistical_ptag(url, int(sys.argv[3]))
            except:
                text = general_statistical_ptag(url, 15)
        elif sys.argv[2] == "general_statistical": 
            try:
                text = general_statistical(url, int(sys.argv[3]))
            except: 
                text = general_statistical(url, 15)
        elif sys.argv[2] == "get_youtube":
            text = get_youtube(url, 10)
        else:
            text = load_article(url)
    else:
        text = load_article(url)

    return text


if __name__ == "__main__":
    text = CLI_run()
    print(text)

    # text = load_article("https://www.rogerebert.com/reviews/you-dont-nomi-movie-review-2020")
    # print(text)
