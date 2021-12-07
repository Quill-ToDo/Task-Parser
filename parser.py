from microtc.utils import tweet_iterator
from os.path import join
import spacy
import json

FILE = "tasks.json"
    
def validate(input, output, total_inputs):
    '''
    Visualize differences between input and output dataset and output in "differences.json"
    '''

    differences = []

    for i in range(len(input)):
        input_task = input[i]
        output_task = output[i]
        original = input_task["input"]
        del input_task["input"]
        if input_task != output_task:
            differences.append({"original input": original, "correct groups": input_task, "our output": output_task})
    
    with open("differences.json", "w") as f:
        json.dump(differences, f, indent=4, separators=(', ', ': '))
    if len(differences) != 0:
        print("There were", str(len(differences)) + "/" + str(total_inputs), "different outputs between the input and output files, check differences.json")

def format_answers(answers):
    '''
    Format outputs for validate()
    '''
    if len(answers["task"]) != 0:
        answers["task"] = " ".join(answers["task"])
    else: 
        answers["task"] = None
    if len(answers["date"]) == 0:
        answers["date"] = None
    else:
        answers["date"] = " ".join(answers["date"])
    if len(answers["recurrence"]) == 0:
        answers["recurrence"] = None

@spacy.Language.component(
    "get_recurrence_entities",
    retokenizes=True
)
def get_recurrence_entities(doc):
    if "every" in doc.text:
        with doc.retokenize() as retokenizer:
            for i, token in enumerate(doc):
                if token.text == "every":
                    start = i
                    end = len(doc)
                    recurrences = spacy.tokens.Span(doc, start, end, label="RECURRENCE")
                    retokenizer.merge(recurrences, attrs={"ent_type": 7884667884033787756})
                    break
    return doc

def is_group(np):
    for token in np:
        if token.ent_type_ != "GROUP":
            return False 
    return True

def does_not_contain_group(np):
    for token in np:
        if token.ent_type_ == "GROUP":
            return False 
    return True

@spacy.Language.component(
    "merge_nouns_without_group",
    requires=["token.dep", "token.tag", "token.pos"],
    retokenizes=True
)
def merge_nouns_without_group(doc):
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

def get_nlp(exclude_list, groups):
    entity_patterns = []
    for group in groups:
        # if the lowercase version of the token matches our word then add it
        p = [{"LOWER": word.lower()} for word in group.split(" ")] 
        ep = {"label": "GROUP", "pattern": p}
        entity_patterns.append(ep)

    nlp = spacy.load("en_core_web_sm", exclude=exclude_list)
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"}, after="ner")
    ruler.add_patterns(entity_patterns)
    nlp.add_pipe("get_recurrence_entities")
    nlp.add_pipe("merge_nouns_without_group", after="get_recurrence_entities")
    return nlp

def is_date_or_time(token):
    return token.ent_type_ == "DATE" or token.ent_type_ == "TIME"

def include_in_task(token):
    ADP_before_date = token.i + 1 < len(token.doc) and token.pos_ == "ADP" and is_date_or_time(token.nbor())
    in_included_pos = token.pos_ == "VERB" \
                    or token.pos_ == "ADJ" \
                    or token.pos_ == "AUX" \
                    or token.pos_ == "NOUN" \
                    or token.pos_ == "PROPN" \
                    or token.pos_ == "ADP" \
                    or token.pos_ == "ADV" \
                    or token.pos_ == "DET" \
                    or token.pos_ == "PART" \
                    or token.pos_ == "PUNCT" \
                    or token.pos_ == "INTJ"
    
    # return in_included_pos and not (ADP_before_date or is_date_or_time(token) or token.ent_type_ == "RECURRENCE")
    return in_included_pos and not (ADP_before_date)

def attached_to_last_word(token):
    '''
    True if token should be appended to the last token
    (Should attach to last word if it's a contraction or punctuation)
    '''
    # Includes things like "n't" and "to"
    return (token.pos_ == "PART" and "'" in token.text) or token.pos_ == "PUNCT"

def parse_body(doc, answers):
    for token in doc:
        if token.ent_type_ == "GROUP":
            answers["group"] = token.text
        
        if token.ent_type_ == "RECURRENCE":
            answers["recurrence"] = token.text
        elif token.ent_type_ == "DATE":
            answers["date"].append(token.text)
        elif token.ent_type_ == "TIME":
            answers["time"] = token.text
        elif include_in_task(token):
            if attached_to_last_word(token):
                answers["task"][-1] += token.text
            else:
                answers["task"].append(token.text)

if __name__ == "__main__":
    # !!!Make sure you run this: $ python -m spacy download en_core_web_sm
    dataset = json.load(open(FILE))

    # These will be set by the user.
    predefined_groups = ["Bio", "Cosc", "Computer Science", "Japanese", "English"]

    # Pipes we don't need
    exclude_list = [
        "DependencyParser",
        "EntityLinker",
        "Morphologizer",
        "SentenceRecognizer",
        "Sentencizer", 
        "TextCategorizer",
        "Tok2Vec",
        "TrainablePipe",
        "Transformer"]

    nlp = get_nlp(exclude_list, predefined_groups)

    results = []

    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = { "group": None, "task": [], "date": [], "time": None, "recurrence": [] }

        parse_body(doc, answers)

        format_answers(answers)
        
        results.append(answers)
    
    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f, indent=4, separators=(', ', ': '))
        
    validate(dataset, results, total_inputs=len(dataset))