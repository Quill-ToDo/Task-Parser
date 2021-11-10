from microtc.utils import tweet_iterator
from os.path import join
import spacy
import json

FILE = "tasks.json"

def validate(input, output):
    '''
    Visualize differences between input and output dataset
    '''
    pass

if __name__ == "__main__":
    # !!!Make sure you run this: $ python -m spacy download en_core_web_sm
    
    dataset = json.load(open(FILE))

    nlp = spacy.load("en_core_web_sm")

    predefined_groups = {"bio", "cosc", "computer science"}
    
    results = []
    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = {}
        answers["task"] = []

        for word in doc:
            # check for acronyms before this because something like bio is recognized as an adjective
            # check if it's in a group before these:
            if word.text in predefined_groups:
                # Must be checked separately because these group names could be nouns, adjectives, etc.
                answers["group"] = word.text
                if word.pos_ == "VERB" or word.pos_ == "NOUN" or word.pos_ == "PROPN": 
                    # Must come after date/time because dates are proper noun
                    answers["task"].append(word.text)

            if word.ent_type_ == "DATE":
                answers["date"] = word.text
            elif word.ent_type == "TIME":
                answers["time"] = word.text
            elif word.pos_ == "VERB" or word.pos_ == "NOUN" or word.pos_ == "PROPN": 
                # Must come after date/time because dates are proper noun
                answers["task"].append(word.text)
        
        answers["task"] = " ".join(answers["task"])
        results.append(answers)
    
    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f)


    # print(word.text, "pos:", word.pos_, "tag:", word.tag_, "ent type:", word.ent_type_)




