from microtc.utils import tweet_iterator
from os.path import join
from datetime import datetime
import spacy
import json
import parsedatetime
import holidays
import re
from collections import Counter

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
    if answers["task"]:
        answers["task"] = " ".join(answers["task"])
    else: 
        answers["task"] = None

    if len(answers["datetime"]) == 0:
        answers["datetime"] = None
    else:
        answers["datetime"] = " ".join(answers["datetime"])

    if not answers["recurrence"]:
        answers["recurrence"] = None
    
    if answers["group"]:
        answers["group"] = list(answers["group"])
        answers["group"] = None

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

def get_nlp(exclude_list, groups, holidays_set):
    entity_patterns = []
    for group in groups:
        # if the lowercase version of the token matches our word then add it
        p = [{"LOWER": word.lower()} for word in group.split(" ")] 
        ep = {"label": "GROUP", "pattern": p}
        entity_patterns.append(ep)
    for holiday in holidays_set:
        # if the lowercase version of the token matches our word then add it
        p = [{"LOWER": word.lower()} for word in holiday.split(" ")] 
        ep = {"label": "HOLIDAY", "pattern": p}
        entity_patterns.append(ep)

    nlp = spacy.load("en_core_web_sm", exclude=exclude_list)

    # Set ER to assign our groups over other entity types
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"})
    entity_patterns2 = []
    for holiday in holidays_set:
        # if the lowercase version of the token matches our word then add it
        p = [{"LOWER": word.lower()} for word in holiday.split(" ")] 
        ep = {"label": "HOLIDAY", "pattern": p}
        entity_patterns2.append(ep)

    # figure out duration parsing somewhere in here

    # Set ER to assign our groups over other entity types
    ruler.add_patterns(entity_patterns)
    nlp.add_pipe("get_recurrence_entities")
    nlp.add_pipe("merge_nouns_without_group", after="get_recurrence_entities")
    return nlp

def is_date_or_time(token):
    #HOLIDAY ent_type_ does not work, it appears as if it is never assigned
    return token.ent_type_ == "DATE" or token.ent_type_ == "TIME" or token.ent_type_ == "HOLIDAY"

#includes almost all words in task unless they're a date or time or adposition before date or time
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
                    or token.pos_ == "INTJ" \
                    or token.pos_ == "PRON"
    
    # return in_included_pos and not (ADP_before_date or is_date_or_time(token) or token.ent_type_ == "RECURRENCE")
    return in_included_pos and not (ADP_before_date)

def attached_to_last_word(token):
    '''
    True if token should be appended to the last token
    (Should attach to last word if it's a contraction or punctuation)
    '''
    # Includes things like "n't" and "to"
    return (token.pos_ == "PART" and "'" in token.text) or token.pos_ == "PUNCT"

# def get_holidays():
#     all_holidays = holidays.US()
#     country_list = re.findall(r"\b[A-Z][a-z\W]*?\b", ' '.join(holidays.list_supported_countries()))
#     i = 0
#     for country in country_list:
#         all_holidays += holidays.CountryHoliday(country, years=datetime.now().year)
#         print(i)
#         i += 1
#     holiday_dict = {}
#     for date, name in sorted(all_holidays.items()):
#         holiday_dict[name] = str(date).split("-", 1)[1]
#     with open("holiday_list.json", "w") as f:
#         json.dump(holiday_dict, f, indent=4, separators=(', ', ': '))

def parse_body(doc, answers):
    for token in doc:
        
        if token.ent_type_ == "RECURRENCE":
            answers["recurrence"] = token.text
        elif token.ent_type_ == "DATE" or token.ent_type_ == "ORDINAL" or token.ent_type_ == "TIME":
                p = parsedatetime.Calendar()
                if token.ent_type_ == "ORDINAL":
                    is_noun = token.end < len(doc) \
                            and (doc[token.end].pos_ == 'NOUN' \
                                or doc[token.end].pos_ == 'PROPN' \
                                or doc[token.end].pos_ == 'ADJ' \
                                or (doc[token.end].pos_ == 'ADP' \
                                    and token.end + 1 < len(doc) \
                                    and token.end + 1 in doc.ents \
                                    and token.end + 1 ))
                    
                    # if ent.end < len(doc) and doc[ent.end].pos_ != 'NOUN':
                    #     print()
                answers["datetime"].append(p.parseDT(doc.text)[0].strftime("%m/%d/%y %H:%M"))
        elif include_in_task(token):
            if attached_to_last_word(token):
                answers["task"][-1] += token.text
            else:
                answers["task"].append(token.text)

def groups_from_acronyms(input, abbrev_dict):
    '''
    Finds acronyms or abbreviations for a group name in the user input task
    '''
    abbrev = re.compile("[a-zA-Z]{2,}")
    output = abbrev.findall(input)
    entities = Counter(output)
    found_groups = set()
    
    for key in entities.keys():
        key = str(key).lower()
        for group in predefined_groups:
            # check if it's an acronym or if we have already seen it
            if key in abbrev_dict.get(group):
                found_groups.add(group)
            # check if it's an abbreviation of a group name
            if key[0] == group[0].lower() and key in group.lower():
                abbrev_dict[key] = abbrev_dict.get(group).add(key)
                found_groups.add(group)
    return found_groups

def add_acronyms(groups, abbrev_dict):
    for group in groups:
        group_terms = group.split(" ")
        if len(group_terms) > 1:
            acronym = ""
            for t in group_terms:
                acronym += t[0]
            abbrev_dict[group].add(acronym.lower())

if __name__ == "__main__":
    # !!!Make sure you run this: $ python -m spacy download en_core_web_sm

    dataset = json.load(open(FILE))

    # These will be set by the user.
    predefined_groups = ["Biology", "Computer Science", "Japanese", "English"]
    predefined_groups.sort()
    holidays_set = ["Christmas", "Valentine's Day", "Halloween", "Easter", "Passover", "Hanukkah", "Chanukah", "New Year's Eve", "New Year's Day", "Diwali", "Eid al-Fitr",
            "Saint Patrick's Day", "Thanksgiving"]

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

    nlp = get_nlp(exclude_list, predefined_groups, holidays_set)
    abbrev_dict = {group : set() for group in predefined_groups} # keep track of all abbreviations for group names that we have seen
    add_acronyms(predefined_groups, abbrev_dict)

    results = []

    # get_holidays()

    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = { "group": set(), "task": [], "datetime": [], "recurrence": [] }

        parse_body(doc, answers)

        answers["group"].update(groups_from_acronyms(input_task, abbrev_dict))

        format_answers(answers)
        
        results.append(answers)

    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f, indent=4, separators=(', ', ': '))
        
    validate(dataset, results, total_inputs=len(dataset))
