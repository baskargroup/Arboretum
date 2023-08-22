import nltk
from tqdm import tqdm
from itertools import chain
from nltk.corpus import stopwords, wordnet
try:
    from fuzzywuzzy import fuzz
except:
    print("Fuzzy matching not available")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_captions(x):
    import logging
    try:
        cleaned = str(x).lower().translate({ord(i): None for i in '&<^*>\\|+=[]~`\"@/\'\©#)("'}).translate({ord(i): " " for i in ':;-_.,!?\n'})\
        .replace(" www ", " ").replace(" com ", " ")\
        .replace(" flickr ", " ").replace(" st ", " street ").replace(" de ", "")\
        .replace("http", " ").replace("href", " ")\
        .strip()
        c_list = list(cleaned.split(" "))
        c_list = [c for c in c_list if (len(c) < 30 and not c.isnumeric())]
        # if len(c_list) > 50:
        #     c_list = c_list[:49]
        return " ".join(c_list)
    except Exception as e:
        logging.info("Exception in clean captions: ")
        logging.info(e)
        return ""

def list_clean_nosplit(l):
    return set(map(clean_captions, l))

def list_clean(l):
    l = list(map(clean_captions, l))
    s = set(chain(*[s.split(" ") for s in l]))
    k = [c for c in s if c not in common]
    return k

def reverse_words(s):
    return "".join(reversed(s.split()))

def synsets(s):
    d = wordnet.synsets(s)
    retlist = []
    if d == []:
        return retlist
    else:
        for ss in d:
            if ss.pos() == 'n':
                m = list(chain(ss.hyponyms() + ss.hypernyms() + ss.also_sees() + ss.similar_tos()))
                l = list(s.replace("_", " ") for s in chain(*list(k.lemma_names() for k in m)))
                retlist = retlist + l
    return retlist

def ds_val_getter(ds, synset):
    if isinstance(ds, dict):
        keys = list(ds.keys())
        altvals = [[] for i in range(len(keys))]
        if synset:
            vals = list(ds.values())
            tlist = [[] for _ in range(len(keys))]
            for i in range(len(keys)):
                tlist[i] = [t.lower().strip() for t in vals[i].split(", ")]
            for slot, t in enumerate(tlist):
                for idx, val in enumerate(t):
                    altvals[slot] = altvals[slot] + list(chain(synsets(t[idx])))
        ds_values = {idx:list(set(altvals[idx] + [t.lower().strip() for t in row.split(", ")] + [reverse_words(t.lower().strip()) for t in row.split(", ")] + [t.lower().strip()+"s" for t in row.split(", ")] + [t.lower().strip().replace(" ", "") for t in row.split(", ")] + [reverse_words(t.lower().strip()).replace(" ", "") for t in row.split(", ")])) for idx, row in enumerate(ds.values())}
    else:
        if synset:
            ds_values = {idx:list(set([ds[idx].lower().strip(), reverse_words(ds[idx].lower().strip()), ds[idx].lower().strip()+"s", ds[idx].lower().strip().replace(" ", ""), reverse_words(ds[idx].lower().strip()).replace(" ", "")] + list(chain(synsets(ds[idx].lower().strip()))))) for idx in range(len(ds))}
        else:
            ds_values = {idx:list(set([ds[idx].lower().strip(), reverse_words(ds[idx].lower().strip()), ds[idx].lower().strip()+"s", ds[idx].lower().strip().replace(" ", ""), reverse_words(ds[idx].lower().strip()).replace(" ", "")])) for idx in range(len(ds))}
    return ds_values

def in1k_hard_subset_match(s, ds, ngram=3, multiclass=False, strict=False, fuzzy=0, synset=False):
    s = str(s)
    if len(s) < 5:
        return -1
    ds_values = ds_val_getter(ds, synset)
    s = list(lemmatizer.lemmatize(t) for t in s.split(" "))
    grams = []
    for count, word in enumerate(s):
        for i in range(ngram):
            if count + i - 1 > len(s):
                continue
            grams.append(" ".join(w for w in s[count:count+i+1]))
            
    matches = []
    
    for gram in grams:
        for idx, val in enumerate(ds_values.values()):
            if gram in val:
                if multiclass or strict:
                    matches.append(idx)
                else:
                    return idx
            elif fuzzy > 0:
                rat = fuzz.ratio(gram,val)
                if rat > fuzzy:
                    matches.append(idx)   
    if matches == []:
        return -1
    elif strict and len(matches) != 1:
        return -1
    elif multiclass:
        matches = set(matches)
        return ", ".join(str(m) for m in matches)
    else:
        return matches[0]


def matched_multi(idx_list, df):
    count = 0
    matchlist = []
    for idx, val in tqdm(enumerate(idx_list)):
        if val == "" or val == "nan":
            matchlist.append(False)
            continue
        if idx > len(df)-1:
            #print(idx)
            break
        if "," in val:
            val = val.split(", ")
            for v in val:
                if int(df["idx"].iloc[idx]) == int(v):
                    count += 1
                    flag = True
            flag = False
            matchlist.append(flag)
        else:
            matchlist.append(int(df["idx"].iloc[idx]) == int(val))
    print("Multiple matches count = {}".format(count))
    return matchlist

def multimatcher(mcstr, scstr, strictstr, df):
    #dftagsoursoa = df[~df[mcstr].isnull()]
    dftagsoursoa = df[df[mcstr] != -1]
    idx_list = [str(s) for s in dftagsoursoa[mcstr].tolist()]
    dftagsoursoa["multimatch"] = matched_multi(idx_list, dftagsoursoa)
    dftagsoursoamatch = dftagsoursoa[dftagsoursoa["multimatch"]]
    dftagsoursoasc = df[df[scstr] != -1]
    dftagsoursoascmatch = df[df[scstr] == df["idx"]]
    dftagsoursoastrict = df[df[strictstr] != -1]
    dftagsoursoastrictmatch = df[df[strictstr] == df["idx"]]

    print("Original dataframe is length {}, {} is length {}, , {} is length {}, {} is length {}".format(len(df), mcstr, len(dftagsoursoa), scstr, len(dftagsoursoasc), strictstr, len(dftagsoursoastrict)))
    print("Accuracy for {} is {:0.2f}, accuracy for {} is {:0.2f}, accuracy for {} is {:0.2f}".format(mcstr, len(dftagsoursoamatch) / len(dftagsoursoa), scstr, len(dftagsoursoascmatch) / len(dftagsoursoasc), strictstr, len(dftagsoursoastrictmatch) / len(dftagsoursoastrict)))