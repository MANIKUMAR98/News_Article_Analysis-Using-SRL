
import spacy

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

text = "While Dinesh and Mani Kuma rgrabbed his flashlight. He began running towards the forest."

doc = nlp(text)

# Dictionary to hold actors and actions
actors_actions = {}

def add_action(subjects, action):
    for subject in subjects:
        if subject not in actors_actions:
            actors_actions[subject] = []
        actors_actions[subject].append(action)

# Function to extract full names
def extract_full_names(token):
    full_name = token.text
    # Expand to the left
    left_tokens = [token]
    while left_tokens[-1].i > 0 and doc[left_tokens[-1].i - 1].pos_ == "PROPN":
        left_tokens.append(doc[left_tokens[-1].i - 1])
    # Expand to the right
    right_tokens = [token]
    while right_tokens[-1].i < len(doc) - 1 and doc[right_tokens[-1].i + 1].pos_ == "PROPN":
        right_tokens.append(doc[right_tokens[-1].i + 1])
    # Combine
    full_range = list(set(left_tokens + right_tokens))
    full_range.sort(key=lambda x: x.i)
    full_name = " ".join([token.text for token in full_range])
    return full_name

recent_subjects = []  # List to hold the most recent proper noun subjects

for sent in doc.sents:
    subjects_this_sentence = []  # Track subjects identified in the current sentence
    for token in sent:
        if token.pos_ == "PROPN" and token.dep_ in ["nsubj", "nsubjpass", "conj"]:
            full_name = extract_full_names(token)
            if full_name not in subjects_this_sentence:
                subjects_this_sentence.append(full_name)

        elif token.pos_ == "VERB":
            subjects_for_verb = []

            if any(child.pos_ == "PRON" and child.dep_ in ["nsubj", "nsubjpass"] for child in token.children):
                subjects_for_verb.extend(recent_subjects)
            elif subjects_this_sentence:
                subjects_for_verb.extend(subjects_this_sentence)

            add_action(subjects_for_verb, token.lemma_)

    if subjects_this_sentence:
        recent_subjects = subjects_this_sentence.copy()