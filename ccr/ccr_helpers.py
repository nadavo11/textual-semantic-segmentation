
"""
ccr_helpers.py

Utilities for running CCR (Cultural Correlates of Representations) on arbitrary questionnaires.

Includes:
- encode_column: embed a text column with SBERT
- item_level_ccr: cosine similarities between data and questionnaire items
- ccr_wrapper: pipeline wrapper
- load_glove / _avg_vecs / tokenize / read_dic_file / __load_dictionary / filter_by_embedding_vocab / _dictionary_centers
- count: apply dictionary categories to text
- csv_to_dic: convert CSV wordlists to dictionary file
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

import re
import sys
import string
import ntpath
import random

def encode_column(model, filename, col_name):
    print("📖 read csv")
    df = pd.read_csv(filename)

    df = df.dropna(subset=[col_name])
    print("🧬embedding column...")
    df["embedding"] = list(model.encode(df[col_name],batch_size=2048))
    return df


from tqdm import tqdm
from sentence_transformers import util

def item_level_ccr(data_encoded_df, questionnaire_encoded_df):
    print("🔍 Computing cosine similarities between data and questionnaire embeddings...")

    q_embeddings = questionnaire_encoded_df.embedding
    d_embeddings = data_encoded_df.embedding

    similarities = util.pytorch_cos_sim(d_embeddings, q_embeddings)

    print("📊 Populating similarity scores into DataFrame...")

    for i in tqdm(range(1, len(questionnaire_encoded_df) + 1), desc="Computing sim_item_X columns"):
        data_encoded_df[f"sim_item_{i}"] = similarities[:, i - 1]

    print("✅ Similarity computation complete.")
    return data_encoded_df



def ccr_wrapper(data_file, data_col, q_file, q_col, model='all-MiniLM-L6-v2'):
    """
    Returns a Dataframe that is the content of data_file with one additional column for CCR value per question

    Parameters:
        data_file (str): path to the file containing user text
        data_col (str): column that includes user text
        q_file (str): path to the file containing questionnaires
        q_col (str): column that includes questions
        model (str): name of the SBERT model to use for CCR see https://www.sbert.net/docs/pretrained_models.html for full list

    """
    print("hola mundo")
    try:
        model = SentenceTransformer(model, device="cuda").to('cuda')
        print("model successfully mounted on cuda⚡")
    except:
        model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda").to('cuda')
        print("model successfully mounted on cuda⚡")
    questionnaire_filename = q_file
    data_filename = data_file

    q_encoded_df = encode_column(model, questionnaire_filename, q_col)
    data_encoded_df = encode_column(model, data_filename, data_col)

    ccr_df = item_level_ccr(data_encoded_df, q_encoded_df)
    # ccr_df = ccr_df.drop(columns=["embeddings"])

    # ccr_df.to_csv("ccr_results.csv")
    return ccr_df

def load_glove(path, vocab, embed_size=300):
    E = dict()
    vocab = set(vocab)
    found = list()
    with open(path) as fo:
        for line in fo:
            tokens = line.strip().split()
            vec = tokens[len(tokens) - embed_size:]
            token = "".join(tokens[:len(tokens) - embed_size])
            E[token] = np.array(vec, dtype=np.float32)
            if token in vocab:
                found.append(token)
    if vocab is not None:
        print("Found {}/{} tokens in {}".format(len(found),
                                len(vocab), path))
    return E, embed_size

def _avg_vecs(words, E, embed_size=300, max_size=None, min_size=1, sampling=False, sampling_k=4, verbose=False):
    vecs = list()
    if sampling:
        if len(words)>sampling_k:
            words = random.sample(words, sampling_k)
        else:
            print("ignored sampling for ", words)
        # print(words)
        for w in words:
            vecs.append(E[w])

    else:
        for w in words:
            if w in E:
                vecs.append(E[w])
            if max_size is not None:
                if len(vecs) >= max_size:
                    break
    print(len(vecs))
    # print(words, len(vecs))
    if len(vecs) < min_size:
        empty_array = np.empty((embed_size,))
        empty_array[:] = np.NaN
        return empty_array
    return np.array(vecs).mean(axis=0)

remove = re.compile(r"(?:http(s)?[^\s]+|(pic\.[^s]+)|@[\s]+)")
alpha = re.compile(r'(?:[a-zA-Z\']{2,15}|[aAiI])')
printable = set(string.printable)


def tokenize(t, stem=False):
    t = remove.sub('', t)
    t = "".join([a for a in filter(lambda x: x in printable, t)])
    tokens = alpha.findall(t)
    return tokens

def read_dic_file(f):
    categories = dict()
    words = list()
    with open(f, 'r') as fo:
        for line in fo:
            if line.strip() == '':
                continue
            if line.startswith("%"):
                continue
            line_split = line.split()
            # print(line_split)
            if line_split[0].isnumeric() and len(line_split) == 2:

                cat_id, category = line.split()
                categories[int(cat_id)] = category
            else:
                words.append(line_split)
    dictionary = {category: list() for id_, category in categories.items()}
    for line in words:
        # print(line)
        word = line[000]
        if line[1][0].isalpha():
            continue  # multi word expression
        for cat_id in line[1:]:
            dictionary[categories[int(cat_id)]].append(word)

    return dictionary


def __load_dictionary(dic_file_path):
    # loads words and stems, builds regx: words are put in regx as they are, stems are handled by allowing any char to follow
    d_name = ntpath.basename(dic_file_path).split('.')[0]
    loaded = read_dic_file(dic_file_path)
    words, stems = dict(), dict()
    for cat in loaded:
        words[cat] = list()
        stems[cat] = list()
        for word in loaded[cat]:
            if word.endswith('*'):
                stems[cat].append(word.replace('*', ''))
            else:
                words[cat].append(word)
    rgxs = dict()
    for cat in loaded:
        name = "{}.{}".format(d_name,cat)
        if len(stems[cat]) == 0:
            regex_str = r'\b(?:{})\b'.format("|".join(words[cat]))
        else:
            unformatted = r'(?:\b(?:{})\b|\b(?:{})[a-zA-Z]*\b)'
            regex_str = unformatted.format("|".join(words[cat]),
                    "|".join(stems[cat]))
        rgxs[name] = re.compile(regex_str)
    return rgxs, words, stems

def filter_by_embedding_vocab(E, words):
    filtered_words = []
    oov_words = []
    for word in words:
        if word in E:
            filtered_words.append(word)
        else:
            oov_words.append(word)
    return filtered_words, oov_words

def _dictionary_centers( d_path, d_name, E, vec_size, max_size=25, sampling=False, sampling_k=4, verbose=False):
    _, d_words, _ = __load_dictionary(d_path)


    #filtering the oov words:
    oov_words = {}
    for cat in d_words:
        d_words[cat], oov_words[cat] = filter_by_embedding_vocab(E, d_words[cat])

    # vocab = list(set([w for cat in d_words for w in d_words[cat]]))
    # path = PRETRAINED[vec_name]
    # E, vec_size = load_glove(path, vocab)

    names = list()
    vecs = list()
    # print(d_words)
    for category in d_words:
        to_append_vecs = _avg_vecs(d_words[category],
                E, embed_size=vec_size, max_size=max_size, sampling=sampling, sampling_k=sampling_k, verbose=verbose)
        vecs.append(to_append_vecs)
        names.append("{}.ddr.{}".format(d_name, category))
        # print(category, to_append_vecs, len(to_append_vecs))
    return np.array(vecs, dtype=np.float32), names

def count(dic_file_path, ccr_df):

    vectors = list()
    names = list()
    rgxs, _, _ = __load_dictionary(dic_file_path)
    # print(rgxs.keys())
    # print(rgxs["other_questionnairs.BE"])
    # print("00000"*20)
    # print(rgxs)
    for cat in rgxs:
        #     print(cat, rgxs[cat])
        try:
            bow = CountVectorizer(token_pattern=rgxs[cat]) \
                .fit(ccr_df.text.values)
        except:
            ccr_df["{}.count.{}".format(cat[:cat.find('.')], cat[cat.find('.') + 1:])] = 0
            print("missed", cat)
            continue
        # vocab = bow.get_feature_names()
        X = bow.transform(ccr_df.text.values).sum(axis=1)
        ccr_df["{}.count.{}".format(cat[:cat.find('.')], cat[cat.find('.') + 1:])] = np.squeeze(np.asarray(X))

    return ccr_df

def csv_to_dic(csv_dic_path, result_path):
    textfile = open(result_path, "w")
    textfile.write("% \n")
    dic_df = pd.read_csv(csv_dic_path)
    for i, col in enumerate(dic_df.columns):
        textfile.write(str(i + 1) + "\t" + col + " \n")

    textfile.write("% \n")

    for i, col in enumerate(dic_df.columns):
        for word in dic_df[col].dropna():
            textfile.write(word + " \t " + str(i + 1) + " \n")
    textfile.close()


def _ensure_1based_sim_cols(n):
    """Return the list of per-item sim column names in the 1..n scheme you already use."""
    return [f"sim_item_{i}" for i in range(1, n + 1)]

def add_title_scores(sim_df: pd.DataFrame, questionnaire_df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """
    Given:
      - sim_df: output of item_level_ccr (has sim_item_1..sim_item_N)
      - questionnaire_df: the encoded questionnaire rows (M rows), in the SAME order as used to compute sims
        and containing a 'title' column that names the construct/group for each item.
    Produces:
      - new columns sim_<title> that are the (simple) average across that title's items.
    """
    q = questionnaire_df.reset_index(drop=True)
    if title_col not in q.columns:
        raise ValueError(f"Questionnaire is missing required column '{title_col}'")
    n_items = len(q)
    item_cols = _ensure_1based_sim_cols(n_items)

    # Build: title -> list of corresponding sim_item_k columns
    title_to_cols = {}
    for i in range(n_items):
        t = q.at[i, title_col]
        title_to_cols.setdefault(t, []).append(item_cols[i])

    # Average per title
    for t, cols in title_to_cols.items():
        # Simple unweighted mean; replace with .mean(axis=1, skipna=True) if needed
        sim_df[f"sim_{t}"] = sim_df[cols].mean(axis=1)

    return sim_df


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def compute_ccr_from_embeddings(
    embeddings_df: pd.DataFrame,
    questionnaire_file: str,
    q_col: str,
    model: str = "all-MiniLM-L6-v2",
    title_col: str = "dimension"   # <-- average-by column (e.g., "dimension")
) -> pd.DataFrame:
    """
    Returns `embeddings_df` with one additional column per questionnaire item (sim_item_k),
    and, if available, per-title averages (sim_<title_col>).
    """
    print("starting CCR process...")
    try:
        st_model = SentenceTransformer(model, device="cuda").to('cuda')
        print("model successfully mounted on cuda⚡")
    except:
        st_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda").to('cuda')
        print("model successfully mounted on cuda⚡")

    # Encode questionnaire items (reuse your helper for consistency)
    questionnaire_filename = questionnaire_file
    q_encoded_df = encode_column(st_model, questionnaire_filename, q_col)

    # Item-level similarities (same as your pipeline)
    print("🔍 Computing cosine similarities between data and questionnaire embeddings...")
    embeddings_df = item_level_ccr(embeddings_df, q_encoded_df)

    # NEW: auto-aggregate by dimension (if column exists)
    if title_col in q_encoded_df.columns:
        embeddings_df = add_title_scores(embeddings_df, q_encoded_df, title_col=title_col)
        print(f"🧮 Added grouped averages: sim_<{title_col}>")
    else:
        print(f"(i) No '{title_col}' column in questionnaire; skipping grouped averages.")

    print("✅ CCR computation complete.")
    return embeddings_df



