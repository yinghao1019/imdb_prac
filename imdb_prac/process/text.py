
import string
import spacy
import re

from imdb_prac.settings import ENTITY_MAPPINGS

# build spacy model for procees text
spacy.require_gpu()
en_nlp = spacy.load("en_core_web_md")


# replace html element in context
NonHtml = re.compile(
    r"<[^<]+?>|<!--.*?-->|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
# remove continous punctations
rmpct = re.compile(r"[^\w\.,;\?\!<=>\/$'\s]|[\.,;\?\!<=>\/$']{2,}")


def strip_html(text):
    text = NonHtml.sub("", text)
    return text


def rm_punct(text):
    text = rmpct.sub("", text)
    return text


def nlp_preprocess(text, tokenizer):
    tokens = []
    ents = []
    for t in en_nlp(text):
        if t.text not in string.punctuation:
            words = [t.text.lower()]
            words = tokenizer.encode(words, is_pretokenized=True).tokens
            tokens.extend(words)
            if t.ent_type_:
                ents.extend([t.ent_type_+t.ent_iob_]*len(words))
            else:
                ents.extend(["O"]*len(words))

    assert len(tokens) == len(
        ents), f"The sent len not equal to entites num\n Token len:{len(tokens)},entites num:{len(ents)}"

    return tokens, ents


def nlp_potsprocess(tokens, ents, tokenizer):
    texts = []
    for t, e in zip(tokens, ents):
        texts.append((tokenizer.token_to_id(t), ENTITY_MAPPINGS[e]))
    return texts
