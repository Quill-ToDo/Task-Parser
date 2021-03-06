import spacy

@spacy.Language.component("expand_weekday_dates")
def expand_weekday_dates(doc):
    '''
        Expand date entities to include tokens that are plurals and the lemma is a date.
        Ex: Thursdays
    '''
    nlp = spacy.load("en_core_web_sm")
    l_doc = ' '.join([token.lemma_ for token in doc])
    l_doc = nlp(l_doc)
    orig_ents = list(doc.ents)
    new_dates = []
    for lemma_token in l_doc:
        if lemma_token.ent_type_ == "DATE" and doc[lemma_token.i].ent_type_ != "DATE" \
                and (doc[lemma_token.i].morph.get("Number") \
                and doc[lemma_token.i].morph.get("Number")[0] == 'Plur'):
            span = spacy.tokens.Span(doc, lemma_token.i, lemma_token.i+1, label="DATE")
            new_dates.append(span)
            if doc[lemma_token.i].ent_type_:
                orig_ents.remove(doc[lemma_token.i])
    if new_dates:
        doc.ents = orig_ents + new_dates
    return doc


def recurrence_likely(token):
    '''
    This is likely a recurrence string if token is "every" and the next token is:
        - Ent: DATE (), TIME (evening), ORINDAL (third) or
        - POS: NUM (4), ADJ (14th) or
        - Duration: day, week, second, minute, hour, month, year... etc. 
    Or the token is "on" and the next token is:
        - a plural form of a date
    '''
    try:
        nbor = token.nbor()
    except IndexError:
        return False

    recurrence_likely_on = token.lower_ == "on" and (nbor.morph.get("Number") and nbor.morph.get("Number")[0]  == "Plur") \
        and (nbor.ent_type_ == "DATE")
    if recurrence_likely_on:
        return True
    
    durations = set(["day", "week", "weekday", "weekday", "weekend", "second", "minute", "hour", "month", "year", "monday", "mon", "tuesday", "tues", "tue",
        "wednesday", "wed", "thursday", "thurs", "r", "friday", "fri", "saturday", "sat", "sunday", "sun"])
    ent_types = set(["DATE", "TIME", "ORDINAL"])
    pos = set(["NUM", "ADJ"])
    recurrence_likely_every = (token.text == "every" and (nbor.ent_type_ in ent_types or nbor.pos_ in pos or nbor.lower_ in durations))
    return recurrence_likely_every


@spacy.Language.component(
    "get_recurrence_entities",
    retokenizes=True
)
def get_recurrence_entities(doc):
    '''
        Retokenize to combine tokens indicating recurrence and label as "RECURRENCE"
    '''
    for token in doc:
        if recurrence_likely(token):
            start = token.i
            for ent in doc.ents:
                if (ent.label_ == "TIME" or ent.label_ == "DATE") and (ent.start < start < ent.end):
                    start = ent.start
            end = len(doc)

            with doc.retokenize() as retokenizer:
                recurrences = spacy.tokens.Span(doc, start, end, label="RECURRENCE")
                retokenizer.merge(recurrences, attrs={"ent_type": 7884667884033787756})
            break
    return doc
        

@spacy.Language.component(
    "merge_nouns_without_group",
    requires=["token.dep", "token.tag", "token.pos"],
    retokenizes=True
)
def merge_nouns_without_group(doc):
    '''
        Modification of merge_nouns to only merge the nouns if they do not contain a token with a GROUP
        label 
    '''
    if not doc.has_annotation("DEP"):
        return doc
    with doc.retokenize() as retokenizer:
        for np in doc.noun_chunks:
            if does_not_contain_group(np):
                attrs = {"tag": np.root.tag, "dep": np.root.dep}
                retokenizer.merge(np, attrs=attrs)  # type: ignore[arg-type]
            elif is_group(np):
                retokenizer.merge(np, attrs={"ent_type_": "GROUP"})  # type: ignore[arg-type]
    return doc

def does_not_contain_group(np):
    for token in np:
        if token.ent_type_ == "GROUP":
            return False 
    return True

def is_group(np):
    for token in np:
        if token.ent_type_ != "GROUP":
            return False 
    return True