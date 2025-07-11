import spacy
from keybert import KeyBERT
import json

# 📌 Load spaCy and KeyBERT
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

# 📌 Known asset types for your domain
ASSET_TYPES = {"icon", "logo", "patch", "sticker", "image", "photo", "illustration"}


def extract_main_intent(query):
    # 1️⃣ Get top key phrases from KeyBERT
    keywords = kw_model.extract_keywords(
        query, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=5
    )
    key_phrases = [kw[0] for kw in keywords]

    # 2️⃣ Detect known asset type in full query
    doc = nlp(query)
    asset_type = None
    for token in doc:
        if token.lemma_.lower() in ASSET_TYPES:
            asset_type = token.lemma_.lower()
            break

    # 3️⃣ Parse the top key phrase
    top_phrase = key_phrases[0] if key_phrases else None

    main_intent = None
    sub_intent = None
    modifiers = []

    if top_phrase:
        phrase_doc = nlp(top_phrase)

        # 🔍 Use dependency parsing to handle compound, ROOT, amod
        compound_noun = None
        root_noun = None

        for token in phrase_doc:
            if token.dep_ == "compound" and token.pos_ in ["NOUN", "PROPN"]:
                if token.lemma_.lower() not in ASSET_TYPES:
                    compound_noun = token.text

            if token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]:
                if token.lemma_.lower() not in ASSET_TYPES:
                    root_noun = token.text

        for token in phrase_doc:
            if token.dep_ in ["amod"]:
                modifiers.append(token.text)

        # 🧩 Final logic: prefer compound noun over ROOT for phrases like "tuna meat"
        if compound_noun:
            main_intent = compound_noun
            if root_noun and root_noun != compound_noun:
                sub_intent = root_noun
        elif root_noun:
            main_intent = root_noun

    return main_intent, sub_intent, modifiers, asset_type, keywords


def main():
    query = input("Enter your search query: ").strip()

    if not query:
        print("No input provided. Exiting.")
        return

    main_intent, sub_intent, modifiers, asset_type, keywords = extract_main_intent(
        query
    )

    result = {
        "query": query,
        "main_intent": main_intent,
        "sub_intent": sub_intent,
        "modifiers": modifiers,
        "asset_type": asset_type,
        "top_keywords": keywords,
    }

    print("\n=== Extracted Intent ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
