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
            input_task["input"] = original
            differences.append({"input": input_task, "output": output_task})
    
    with open("differences.json", "w") as f:
        json.dump(differences, f)
    return len(differences) == 0

def include_in_task(word):
    return word.pos_ == "VERB" or word.pos_ == "ADJ" or word.pos_ == "AUX" or word.pos_ == "NOUN" or word.pos_ == "PROPN" or \
            word.pos_ == "ADP"


if __name__ == "__main__":
    # !!!Make sure you run this: $ python -m spacy download en_core_web_sm

    # TODO: 
    # [ ] Read computer science as one group and not two separate things
    # [ ] change input to include group in task
    # [ ] Include auxillary words like do in task  
    # [ ] Only include ADP in task if it is not before a date or in a date
    dataset = json.load(open(FILE))

    nlp = spacy.load("en_core_web_sm")

    predefined_groups = {"bio", "cosc", "computer science"}
    
    results = []
    for data in dataset:
        input_task = data["input"]
        doc = nlp(input_task)
        answers = { "group": None, "task": [], "date": None, "time": None }

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
                answers["date"] = word.text
            elif word.ent_type_ == "TIME":
                answers["time"] = word.text
            elif include_in_task(word): 
                # THIS SHOULD NOT BE CHECKED TWICE BECAUSE THE WORD WILL BE ADDED TWICE
                # Must come after date/time because dates are proper noun
                answers["task"].append(word.text)
        
        answers["task"] = " ".join(answers["task"])
        results.append(answers)
    
    with open("parsed_tasks.json", "w") as f:
        json.dump(results, f)

    if not validate(dataset, results):
        print("There were differences between the input and output files, check differences.json")


    # print(word.text, "pos:", word.pos_, "tag:", word.tag_, "ent type:", word.ent_type_)




