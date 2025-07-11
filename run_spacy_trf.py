import spacy
import json

# 📌 Load spaCy transformer model
nlp = spacy.load("en_core_web_trf")

# 📌 Known asset types
ASSET_TYPES = {
    "icon",
    "logo",
    "patch",
    "sticker",
    "image",
    "photo",
    "illustration",
    "shape",
}


def extract_main_intent(query):
    doc = nlp(query)

    # 1️⃣ Detect asset type
    asset_type = None
    for token in doc:
        if token.lemma_.lower() in ASSET_TYPES:
            asset_type = token.lemma_.lower()
            break

    # 2️⃣ Pick noun chunks
    noun_chunks = list(doc.noun_chunks)
    noun_tokens = [token for token in doc if token.pos_ == "NOUN"]

    candidates = [chunk.text for chunk in noun_chunks] + [
        token.text for token in noun_tokens
    ]

    candidates = sorted(set(candidates), key=lambda x: -len(x))

    main_intent = None
    sub_intent = None
    modifiers = []

    if candidates:
        top_phrase = candidates[0]
        phrase_doc = nlp(top_phrase)

        compound = None
        root = None

        for token in phrase_doc:
            if token.dep_ == "compound":
                compound = token.text
            if token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]:
                root = token.text
            if token.dep_ == "amod":
                modifiers.append(token.text)

        main_intent = compound if compound else root

    return main_intent, sub_intent, modifiers, asset_type, candidates


def main():
    query = input("Enter your search query: ").strip()

    if not query:
        print("No input provided. Exiting.")
        return

    main_intent, sub_intent, modifiers, asset_type, candidates = extract_main_intent(
        query
    )

    result = {
        "query": query,
        "main_intent": main_intent,
        "sub_intent": sub_intent,
        "modifiers": modifiers,
        "asset_type": asset_type,
        "candidates": candidates,
    }

    print("\n=== Extracted Intent (trf) ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
