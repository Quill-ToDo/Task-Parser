from microtc.utils import tweet_iterator
from os.path import join
import spacy
import json

FILE = "tasks.json"

def validate(input, output):
    '''
    Visualize differences between input and output dataset
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
        json.dump(differences, f)
    if len(differences) != 0:
        print("There were", len(differences), "different outputs between the input and output files, check differences.json")

def include_in_task(word):
    return word.pos_ == "VERB" or word.pos_ == "ADJ" or word.pos_ == "AUX" or word.pos_ == "NOUN" or word.pos_ == "PROPN" or \
            word.pos_ == "ADP" or word.pos_ == "ADV"


if __name__ == "__main__":
    # !!!Make sure you run this: $ python -m spacy download en_core_web_sm
    dataset = json.load(open(FILE))

    nlp = spacy.load("en_core_web_sm")

    predefined_groups = {"bio", "cosc", "computer science", "cooking"}
    
    results = []
    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = { "group": None, "task": [], "date": [], "time": None }

        for word in doc:
            # check for acronyms before this because something like bio is recognized as an adjective
            # check if it's in a group before these:
            if word.text in predefined_groups:
                # Must be checked separately because these group names could be nouns, adjectives, etc.
                answers["group"] = word.text
                # if include_in_task(word): 
                #     # Must come after date/time because dates are proper noun
                #     answers["task"].append(word.text)

            if word.ent_type_ == "DATE":
                answers["date"].append(word.text)
            elif word.ent_type_ == "TIME":
                answers["time"] = word.text
            elif include_in_task(word): 
                # THIS SHOULD NOT BE CHECKED TWICE BECAUSE THE WORD WILL BE ADDED TWICE
                # Must come after date/time because dates are proper noun

                answers["task"].append(word.text)
        
        answers["task"] = " ".join(answers["task"])
        answers["date"] = " ".join(answers["date"])
        results.append(answers)
    
    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f)

    validate(dataset, results)



    # print(word.text, "pos:", word.pos_, "tag:", word.tag_, "ent type:", word.ent_type_)




