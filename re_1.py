import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("dataset.csv")

def extract_relations(text):
    doc = nlp(str(text))
    triples = []
    for sent in doc.sents:
        subject = None
        relation = None
        object_ = None
        for token in sent:
            if "subj" in token.dep_:
                subject = token.text
            if token.pos_ == "VERB":
                relation = token.lemma_
            if "obj" in token.dep_:
                object_ = token.text
        if subject and relation and object_:
            triples.append((subject, relation, object_))
    return triples

data["relations"] = data["text"].apply(extract_relations)

print(data.head())

data.to_csv("relations_output.csv", index=False)

print("Relation extraction complete! Results saved to 'relations_output.csv'")
