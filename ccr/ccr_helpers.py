
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
from tqdm import tqdm
import re, sys, string, ntpath, random
from sklearn.feature_extraction.text import CountVectorizer

# --- NEW: long-text support helpers -----------------------------------------
from typing import List, Tuple, Iterable, Callable

def _get_tokenizer(model_name_or_obj):
    """
    Return a HuggingFace tokenizer aligned with the SentenceTransformer.
    Accepts a model name (str) or an instantiated SentenceTransformer.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers is required for token-aware chunking. pip install transformers")

    if isinstance(model_name_or_obj, str):
        return AutoTokenizer.from_pretrained(model_name_or_obj, use_fast=True)
    # SentenceTransformer object
    try:
        # Most SBERT models expose the underlying HF tokenizer this way:
        return model_name_or_obj._first_module().tokenizer
    except Exception:
        # Fallback: try model name field (not always available)
        try:
            return AutoTokenizer.from_pretrained(model_name_or_obj.model_card, use_fast=True)
        except Exception:
            raise RuntimeError("Could not access tokenizer from SentenceTransformer instance.")


def count_tokens(text: str, tokenizer, add_special_tokens: bool = True) -> int:
    """Number of wordpiece tokens for `text` with the given tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _simple_sentence_split(paragraph: str) -> List[str]:
    """
    Lightweight sentence splitter (no NLTK dependency).
    Splits on . ! ? while keeping punctuation attached.
    """
    # Normalize whitespace
    p = re.sub(r'\s+', ' ', paragraph.strip())
    if not p:
        return []
    # Split on sentence terminators followed by space or end
    # Keep the terminator by using a capturing group.
    parts = re.split(r'([.!?])\s+', p)
    if len(parts) == 1:
        return [p]
    sents = []
    # Re-attach punctuation captured in odd positions
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg:
            continue
        if i + 1 < len(parts):
            seg = seg + parts[i + 1]
        sents.append(seg.strip())
    return sents


def _pack_by_token_budget(
    units: List[str],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int = 0
) -> List[str]:
    """
    Pack a list of units (sentences or paragraphs) into chunks that fit in `max_tokens`.
    If a single unit overflows by itself, it will be split greedily by word.
    """
    chunks = []
    cur = []
    cur_count = 0

    def push_current():
        nonlocal cur, cur_count
        if cur:
            chunks.append(' '.join(cur).strip())
            cur, cur_count = [], 0

    for u in units:
        u = u.strip()
        if not u:
            continue
        u_tokens = tokenizer.encode(u, add_special_tokens=False)
        u_len = len(u_tokens)

        # If unit alone is too big, split by words greedily
        if u_len > max_tokens:
            # push whatever accumulated first
            push_current()
            words = u.split()
            left = 0
            while left < len(words):
                # grow window until it exceeds budget
                # binary/linear mix for simplicity
                right = min(len(words), left + 1)
                best_right = right
                while right <= len(words):
                    cand = ' '.join(words[left:right])
                    if len(tokenizer.encode(cand, add_special_tokens=False)) <= max_tokens:
                        best_right = right
                        right += 1
                    else:
                        break
                chunk = ' '.join(words[left:best_right])
                chunks.append(chunk)
                if overlap_tokens > 0 and best_right < len(words):
                    # Construct overlap window from tail tokens of last chunk
                    # (token-level precise overlap is expensive; we approximate with words)
                    pass
                left = best_right
            continue

        # Try adding to current chunk
        new_count = len(tokenizer.encode((' '.join(cur + [u])).strip(), add_special_tokens=False))
        if new_count <= max_tokens:
            cur.append(u)
            cur_count = new_count
        else:
            # Close current and start new
            push_current()
            cur = [u]
            cur_count = u_len

    push_current()

    # Optional sliding-window token overlap between chunks (approximate)
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped = []
        prev_tail = None
        for i, ch in enumerate(chunks):
            if i > 0 and prev_tail:
                merged = (prev_tail + " " + ch).strip()
                # If merging exceeds budget, keep ch as is
                if len(tokenizer.encode(merged, add_special_tokens=False)) <= max_tokens:
                    overlapped.append(merged)
                else:
                    overlapped.append(ch)
            else:
                overlapped.append(ch)
            # compute tail snippet by token count
            toks = tokenizer.encode(ch, add_special_tokens=False)
            if len(toks) > overlap_tokens:
                # get last ~overlap_tokens worth of words (approximate)
                words = ch.split()
                # back off words until ~overlap token budget
                tail = []
                for w in reversed(words):
                    cand = (' '.join([w] + tail)).strip()
                    if len(tokenizer.encode(cand, add_special_tokens=False)) <= overlap_tokens:
                        tail.insert(0, w)
                    else:
                        break
                prev_tail = ' '.join(tail)
            else:
                prev_tail = ch
        chunks = overlapped

    return chunks


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 384,
    overlap_tokens: int = 32
) -> List[str]:
    """
    Paragraph-first chunking; if a paragraph still overflows, split to sentences; if still
    overflowing, fallback to greedy word packing.
    Returns a list of chunks that each fit in `max_tokens`.
    """
    # Fast path
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text.strip()]

    # 1) Split by blank lines into paragraphs
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks: List[str] = []

    for p in paras:
        if count_tokens(p, tokenizer) <= max_tokens:
            chunks.append(p)
            continue

        # 2) paragraph too long -> split to sentences
        sents = _simple_sentence_split(p)
        if sents and all(count_tokens(s, tokenizer) <= max_tokens for s in sents):
            chunks.extend(_pack_by_token_budget(sents, tokenizer, max_tokens, overlap_tokens))
        else:
            # 3) extreme case: sentence still too long -> greedy word packing
            chunks.extend(_pack_by_token_budget([p], tokenizer, max_tokens, overlap_tokens))

    # Final safety: if anything still too long (shouldn’t happen), truncate last-resort
    safe = []
    for ch in chunks:
        if count_tokens(ch, tokenizer) <= max_tokens:
            safe.append(ch)
        else:
            # truncate as a last resort (logically should not be reached)
            ids = tokenizer.encode(ch, add_special_tokens=False)[:max_tokens]
            safe.append(tokenizer.decode(ids))
    return safe


def _pool_vectors(
    vecs: Iterable[np.ndarray],
    weights: Iterable[float] = None,
    method: str = "mean"
) -> np.ndarray:
    """
    Pool a list of vectors into one.
    method: 'mean' | 'length' (length-weighted mean by token count)
    """
    arr = np.vstack(vecs)
    if method == "length" and weights is not None:
        w = np.array(list(weights), dtype=np.float32).reshape(-1, 1)
        w = w / (w.sum() + 1e-9)
        return (arr * w).sum(axis=0)
    return arr.mean(axis=0)


def embed_long_text(
    model: SentenceTransformer,
    tokenizer,
    text: str,
    max_tokens: int = 384,
    overlap_tokens: int = 32,
    pooling: str = "mean"
) -> np.ndarray:
    """
    Token-aware embedding for long documents:
    - chunk by token budget
    - encode each chunk
    - pool back to a single vector
    """
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    # Track lengths to optionally weight pooling
    lengths = [count_tokens(c, tokenizer, add_special_tokens=False) for c in chunks]
    embs = model.encode(chunks, batch_size=min(64, len(chunks)) if len(chunks) > 0 else 1, convert_to_numpy=True)
    if isinstance(embs, list):
        embs = np.vstack(embs)
    if pooling == "length":
        return _pool_vectors(embs, weights=lengths, method="length")
    return _pool_vectors(embs, method="mean")
# --- end NEW helpers ---------------------------------------------------------


def encode_column(model, filename, col_name,
                  tokenizer=None,
                  max_tokens=384,
                  overlap_tokens=32,
                  pooling="mean",
                  warn_on_truncation=True):
    """
    Read CSV, drop NA in col_name, and add embeddings as 'embedding'.
    For long texts, uses token-aware chunking+pooling to avoid truncation.

    pooling: 'mean' or 'length' (token-length-weighted mean)
    """
    df = pd.read_csv(filename).dropna(subset=[col_name])

    if tokenizer is None:
        # Try to infer tokenizer from model
        tokenizer = _get_tokenizer(model)

    # Ensure SentenceTransformer won't pre-truncate too aggressively
    # (underlying HF model usually caps at 512 anyway)
    try:
        model.max_seq_length = max(model.max_seq_length, max_tokens)
    except Exception:
        pass

    # Vectorize row-wise (handles long texts)
    embs = []
    truncated_count = 0
    for txt in df[col_name].astype(str).tolist():
        tok_count = count_tokens(txt, tokenizer)
        if tok_count > max_tokens:
            truncated_count += 1
            v = embed_long_text(model, tokenizer, txt,
                                max_tokens=max_tokens,
                                overlap_tokens=overlap_tokens,
                                pooling=pooling)
        else:
            v = model.encode([txt], convert_to_numpy=True)[0]
        embs.append(v)

    if warn_on_truncation and truncated_count > 0:
        print(f"⚠️  {truncated_count} / {len(df)} texts exceeded {max_tokens} tokens "
              f"and were chunked (no truncation).")

    df["embedding"] = embs
    return df



def item_level_ccr(data_encoded_df, questionnaire_encoded_df):
    """Compute cosine similarities between data and questionnaire embeddings."""
    q_embeddings = questionnaire_encoded_df.embedding
    d_embeddings = data_encoded_df.embedding
    similarities = util.pytorch_cos_sim(d_embeddings, q_embeddings)
    for i in tqdm(range(1, len(questionnaire_encoded_df) + 1), desc="Computing sim_item_X"):
        data_encoded_df[f"sim_item_{i}"] = similarities[:, i - 1]
    return data_encoded_df


def ccr_wrapper(data_file, data_col, q_file, q_col,
                model='all-MiniLM-L6-v2',
                max_tokens=384,          # <= you can raise to 512 for MiniLM; or 1024+ for longer-context models
                overlap_tokens=32,
                pooling='mean',
                device='cuda'):
    """
    Run full CCR pipeline with token-aware chunking for long texts.

    Parameters
    ----------
    model : str | SentenceTransformer
        Name or instance. Swap to a long-context embedding model if available.
    max_tokens : int
        Token budget per chunk (<= model's real max, commonly 512 for SBERTs).
    overlap_tokens : int
        Sliding-window overlap (token-level, approximate).
    pooling : str
        'mean' or 'length' (length-weighted mean across chunks)
    device : str
        'cuda' or 'cpu'
    """
    # Build/normalize model
    if isinstance(model, str):
        try:
            model = SentenceTransformer(model, device=device).to(device)
        except Exception:
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device).to(device)

    tokenizer = _get_tokenizer(model)

    # Encode questionnaire items (almost always short; no need to chunk)
    q_encoded_df = encode_column(model, q_file, q_col,
                                 tokenizer=tokenizer,
                                 max_tokens=max_tokens,
                                 overlap_tokens=overlap_tokens,
                                 pooling=pooling,
                                 warn_on_truncation=False)

    # Encode data (may be long; token-aware)
    d_encoded_df = encode_column(model, data_file, data_col,
                                 tokenizer=tokenizer,
                                 max_tokens=max_tokens,
                                 overlap_tokens=overlap_tokens,
                                 pooling=pooling,
                                 warn_on_truncation=True)

    return item_level_ccr(d_encoded_df, q_encoded_df)



def load_glove(path, vocab, embed_size=300):
    """Load GloVe vectors for vocab."""
    E = dict()
    vocab = set(vocab)
    found = []
    with open(path) as fo:
        for line in fo:
            tokens = line.strip().split()
            vec = tokens[-embed_size:]
            token = "".join(tokens[:-embed_size])
            E[token] = np.array(vec, dtype=np.float32)
            if token in vocab:
                found.append(token)
    if vocab:
        print(f"Found {len(found)}/{len(vocab)} tokens in {path}")
    return E, embed_size


def _avg_vecs(words, E, embed_size=300, max_size=None, min_size=1, sampling=False, sampling_k=4):
    vecs = []
    if sampling:
        words = random.sample(words, min(len(words), sampling_k))
    for w in words:
        if w in E:
            vecs.append(E[w])
        if max_size and len(vecs) >= max_size:
            break
    if len(vecs) < min_size:
        return np.full((embed_size,), np.nan)
    return np.array(vecs).mean(axis=0)


remove = re.compile(r"(?:http(s)?[^\s]+|(pic\.[^s]+)|@[\s]+)")
alpha = re.compile(r"(?:[a-zA-Z']{2,15}|[aAiI])")
printable = set(string.printable)


def tokenize(t):
    """Tokenize text into alphabetic tokens."""
    t = remove.sub('', t)
    t = "".join([a for a in filter(lambda x: x in printable, t)])
    return alpha.findall(t)


def read_dic_file(f):
    """Read dictionary .dic format file -> {category: words}."""
    categories, words = {}, []
    with open(f, 'r') as fo:
        for line in fo:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            if parts[0].isnumeric() and len(parts) == 2:
                categories[int(parts[0])] = parts[1]
            else:
                words.append(parts)
    dictionary = {cat: [] for cat in categories.values()}
    for line in words:
        word = line[0]
        if line[1][0].isalpha():
            continue
        for cat_id in line[1:]:
            dictionary[categories[int(cat_id)]].append(word)
    return dictionary


def __load_dictionary(dic_file_path):
    """Load .dic file -> regex matchers."""
    d_name = ntpath.basename(dic_file_path).split('.')[0]
    loaded = read_dic_file(dic_file_path)
    words, stems = {}, {}
    for cat in loaded:
        words[cat], stems[cat] = [], []
        for word in loaded[cat]:
            if word.endswith('*'):
                stems[cat].append(word[:-1])
            else:
                words[cat].append(word)
    rgxs = {}
    for cat in loaded:
        name = f"{d_name}.{cat}"
        if stems[cat]:
            regex_str = rf"(?:\b(?:{'|'.join(words[cat])})\b|\b(?:{'|'.join(stems[cat])})[a-zA-Z]*\b)"
        else:
            regex_str = rf"\b(?:{'|'.join(words[cat])})\b"
        rgxs[name] = re.compile(regex_str)
    return rgxs, words, stems


def filter_by_embedding_vocab(E, words):
    return [w for w in words if w in E], [w for w in words if w not in E]


def _dictionary_centers(d_path, d_name, E, vec_size, max_size=25, sampling=False, sampling_k=4):
    _, d_words, _ = __load_dictionary(d_path)
    for cat in d_words:
        d_words[cat], _ = filter_by_embedding_vocab(E, d_words[cat])
    names, vecs = [], []
    for cat in d_words:
        v = _avg_vecs(d_words[cat], E, vec_size, max_size=max_size, sampling=sampling, sampling_k=sampling_k)
        vecs.append(v)
        names.append(f"{d_name}.ddr.{cat}")
    return np.array(vecs, dtype=np.float32), names


def count(dic_file_path, ccr_df):
    """Apply dictionary regex counts to text column 'text'."""
    rgxs, _, _ = __load_dictionary(dic_file_path)
    for cat, regex in rgxs.items():
        try:
            bow = CountVectorizer(token_pattern=regex).fit(ccr_df.text.values)
            X = bow.transform(ccr_df.text.values).sum(axis=1)
            ccr_df[f"{cat}.count"] = np.squeeze(np.asarray(X))
        except:
            ccr_df[f"{cat}.count"] = 0
    return ccr_df


def csv_to_dic(csv_dic_path, result_path):
    """Convert CSV of wordlists -> .dic format file."""
    df = pd.read_csv(csv_dic_path)
    with open(result_path, "w") as fo:
        fo.write("%\n")
        for i, col in enumerate(df.columns, start=1):
            fo.write(f"{i}\t{col}\n")
        fo.write("%\n")
        for i, col in enumerate(df.columns, start=1):
            for word in df[col].dropna():
                fo.write(f"{word}\t{i}\n")
