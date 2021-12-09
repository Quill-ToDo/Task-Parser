from microtc.utils import tweet_iterator
from os.path import join
from datetime import datetime, timedelta
import spacy
import json
import parsedatetime
import re
import copy
import additional_pipelines
from collections import Counter

FILE = "tasks.json"
    
def validate(input, output, p, total_inputs):
    '''
    Visualize differences between input and output dataset and output in "differences.json"
    '''

    differences = []

    for i in range(len(input)):
        input_task = input[i]
        input_task_copy = copy.deepcopy(input[i])
        output_task = output[i]
        original = input_task["input"]

        if input_task["datetime"] != output_task["datetime"]:
            if output_task["datetime"] == p.parseDT(input_task["input"])[0].strftime("%m/%d/%y %H:%M") \
               and output_task["datetime"] != datetime.now().strftime("%m/%d/%y %H:%M"):
                input_task_copy["datetime"] = output_task["datetime"]
        del input_task_copy["input"]
        del input_task["input"]
        if input_task_copy != output_task:
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

    if answers["datetime"]:
        answers["datetime"] = " ".join(answers["datetime"])
    else:
        answers["datetime"] = None

    if not answers["recurrence"]:
        answers["recurrence"] = None
    
    if answers["group"]:
        answers["group"] = list(answers["group"])
    else:
        answers["group"] = None

def get_entity_patterns(groups, holidays):
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
    return entity_patterns

def get_nlp(exclude_list, groups, holidays_set):
    nlp = spacy.load("en_core_web_sm", exclude=exclude_list)
    for holiday in holidays_set:
        # if the lowercase version of the token matches our word then add it
        p = [{"LOWER": word.lower()} for word in holiday.split(" ")] 
        ep = {"label": "HOLIDAY", "pattern": p}
    nlp.add_pipe("expand_weekday_dates")
    # Set ER to assign our labels over other entity types
    nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"}).add_patterns(get_entity_patterns(groups, holidays))
    nlp.add_pipe("get_recurrence_entities", after="entity_ruler")
    nlp.add_pipe("merge_nouns_without_group", after="get_recurrence_entities")
    return nlp

def is_date_or_time(token):
    #HOLIDAY ent_type_ does not work, it appears as if it is never assigned
    return token.ent_type_ == "DATE" or token.ent_type_ == "TIME" or token.ent_type_ == "HOLIDAY"

#includes almost all words in task unless they're a date or time or adposition before date or time
def include_in_task(token):
    ADP_before_removed_portion = token.i + 1 < len(token.doc) and token.pos_ == "ADP" and (is_date_or_time(token.nbor()) or token.nbor().ent_type_ == "RECURRENCE")
    included_pos = set(["VERB", "ADJ", "AUX", "NOUN", "PROPN", "ADP", "ADV", "DET", "PART", "PUNCT", "INTJ", "PRON", "CCONJ"])
    return token.pos_ in included_pos and not (ADP_before_removed_portion)

def attached_to_last_word(token):
    '''
    True if token should be appended to the last token
    (Should attach to last word if it's a contraction or punctuation but NOT if there was a space there in the original task)
    '''
    # Includes things like "n't" and "to"
    return token.idx-1 >= 0 and ((token.pos_ == "PART" and "'" in token.text) or (token.pos_ == "PUNCT" and not doc.text[token.idx-1] == " "))

def parse_body(doc, answers, p, holidays_set):
    for token in doc:
        if token.ent_type_ == "RECURRENCE":
            answers["recurrence"] = token.text
        elif token.ent_type_ == "DATE" or token.ent_type_ == "ORDINAL" or token.ent_type_ == "TIME":
            if answers["datetime"] == []:
                answers["datetime"].append(p.parseDT(doc.text)[0].strftime("%m/%d/%y %H:%M"))
            elif token.ent_type == "DATE":
                answers["datetime"] = p.parseDT(doc.text)[0].strftime("%m/%d/%y %H:%M")
        elif token.ent_type_ == "HOLIDAY":
                if answers["datetime"] == []:
                    answers["datetime"].append(p.parseDT(holidays_set[token.text])[0].strftime("%m/%d/%y %H:%M"))
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
            elif key[0] == group[0].lower() and key in group.lower():
                abbrev_dict.get(group).add(key)
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
    dataset = json.load(open(FILE))

    # These will be set by the user.
    predefined_groups = ["Biology", "Computer Science", "Japanese", "English"]
    predefined_groups.sort()
    holidays_set = {"Christmas": "12/25", "Valentine's Day": "2/14", "Halloween": "10/31", "New Year's Eve": "12/31", "New Year's Day": "1/1",
            "Saint Patrick's Day": "3/17", "Presidents' Day": "2/21", "Fourth of July": "7/4"}

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
    p = parsedatetime.Calendar()
    results = []

    # get_holidays()

    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = { "group": set(), "task": [], "datetime": [], "recurrence": [] }

        parse_body(doc, answers, p, holidays_set)
 
        answers["group"] = groups_from_acronyms(input_task, abbrev_dict)

        format_answers(answers)
        
        results.append(answers)

    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f, indent=4, separators=(', ', ': '))
        
    validate(dataset, results, p, total_inputs=len(dataset))
