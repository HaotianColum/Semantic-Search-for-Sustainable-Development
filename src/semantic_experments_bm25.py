import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd
import os
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
headers = {"user-agent": "js5783@columbia.edu"}
lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word.lower())
        else:
            res_words.append(lemmatizer.lemmatize(word.lower(), tag))
    return " ".join(res_words)

documents_path = 'data/10k_raw/'
form_type = ".txt"
def read_txt(path,file_name):
    text_file = open(f"{path}/{file_name}", "r")
    txt = text_file.read()
    text_file.close()
    return txt

if form_type:
    files = [name for name in os.listdir(f"{documents_path}") if name[-len(form_type):] == form_type]

res_df = pd.DataFrame()
# save to dataframe
for file_name in files:
    cleantext = read_txt(f"{documents_path}", file_name)
    cleantext = lemmatize_sentence(cleantext)
    print(cleantext)
    filing = pd.DataFrame(cleantext, index=[file_name], columns = ['Content'])
    # res_df = pd.concat([res_df, filing], axis=1)
    res_df = res_df.append(filing)

# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
nlp = spacy.load("en_core_web_sm")
text_list = res_df.Content.str.lower().values
sent_list = []
for txt in text_list:
    sents = [txt for txt in sent_tokenize(txt) if len(txt) >= 10]
    sent_list.extend(sents)
tok_text=[] # for our tokenised corpus
#Tokenising using SpaCy:
for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser","ner"])):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)


bm25 = BM25Okapi(tok_text)

query = "emission"
tokenized_query = query.lower().split(" ")
import time
t0 = time.time()
results = bm25.get_top_n(tokenized_query, sent_list, n=3)
t1 = time.time()
print(f'Searched 50,000 records in {round(t1-t0,3) } seconds \n')
for i in results:
    print(i)
