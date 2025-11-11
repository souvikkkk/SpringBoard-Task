import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("dataset.csv")

def extract_entities(text):
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

data["entities"] = data["text"].apply(extract_entities)

persons = set()
orgs = set()
dates = set()

for ents in data["entities"]:
    for ent_text, ent_label in ents:
        if ent_label == "PERSON":
            persons.add(ent_text)
        elif ent_label == "ORG":
            orgs.add(ent_text)
        elif ent_label == "DATE":
            dates.add(ent_text)

print("NER Entity Summary:")
print(f"Unique PERSON entities: {len(persons)}")
print(f"Unique ORG entities: {len(orgs)}")
print(f"Unique DATE entities: {len(dates)}")

print("\nSample PERSONs:", list(persons)[:5])
print("Sample ORGs:", list(orgs)[:5])
print("Sample DATEs:", list(dates)[:5])
