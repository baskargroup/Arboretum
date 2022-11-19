"""
Synset builder (string -> multi-type) takes as input a text string which is assumed to be a dataset caption, and produces as output a transformed caption.
The nature of the transformation depends on the arguments passed to the function.
The primary function of synset_ds is SUBSET MATCHING: matching caption strings to integer-valued classes.
ds: expects a formatted list of classnames or a dict in which the keys are integer values and the values are lists of strings representing classnames. 
ds is the set of terms which are used for subset matching. The terms are task-dependent and will vary for different evaluation datasets.
strict: if this is set to True, then subset matching will be strict, following the methodology of Fang et al ... multiple matches -> no match
cipher: if this is set to a non-zero integer value, then a Caesar cipher will be applied to the input string for that integer value.
fuzzy: Levenstein distance fuzzy matching instead of strict 1 to 1 subset matching
nva uses parts of speech for all of wordnet, instead of matching on some list from a dataset
"""

def synset_ds(s, synset_reqd=False, strict=False, ngram=3, nva=[], ds=None, cipher=False, simplecaptions=False, simplercaptions=False, fuzzy=0):
    ds_dict = {idx:[t.lower().strip() for t in row.split(", ")] for idx, row in enumerate(ds.values())}
    ds_values = set(chain(*[v for v in ds_dict.values()]))
    flag = False
    s = [lemmatizer.lemmatize(t) for t in s.split(" ")]
    str_s = " ".join(w for w in s)
    synset = []
    for count, word in enumerate(s):
        grams = []
        for i in range(ngram):
            if count + i - 1 > len(s):
                continue
            grams.append(" ".join(w for w in s[count:count+i+1]))
        for i, gram in enumerate(grams):
            d = wordnet.synsets(gram)
            if not d and synset_reqd:
                continue
            gram_t = gram
            if cipher:
                k = ""
                for c in gram:
                    nextc = cipher_dict.get(c) or c
                    k = k + nextc
                gram_t = k
            if nva:
                pos = nva
                if gram in stopwords.words('english'):
                    continue
                if d and any(d[k].pos() in pos for k in range(len(d))):
                    synset.append((gram, d))
            if fuzzy > 0:
                fuzzmatch = False
                fuzzes = [(fuzz.ratio(gram_t,d), gram_t, d) for d in ds]
                for f, g, d in fuzzes:
                    if f > fuzzy:
                        fuzzmatch = True
            if (ds is not None and gram_t in ds_values) or (fuzzy > 0 and fuzzmatch):
                if flag and any([strict, simplercaptions]):
                    continue
                if any([simplecaptions, simplercaptions]) and not flag:
                    str_s = "An image of " + gram
                elif simplecaptions and flag and str_s.find(gram) == -1:
                    str_s += " {}".format(gram)
                flag = True
                if cipher:
                    str_s = str_s.replace(gram, k)
            elif simplecaptions and d and any(d[k].pos() in ['n', 'a'] for k in range(len(d))) and d not in stops and not ds and len(gram) > 2:
                if not flag:
                    str_s = "An image of " + gram
                    flag = True
                elif str_s.find(gram) == -1:
                    str_s += " {}".format(gram)
                # print("after simplecaptions, {}".format(len(str_s)))
    if cipher or any([simplecaptions, simplercaptions]):
        if not flag:
            str_s = ""
        if flag:
            return str_s
    return flag