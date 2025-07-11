import spacy
import json

# 📌 Load spaCy large model
nlp = spacy.load("en_core_web_lg")

# 📌 Attribute categories
ATTRIBUTE_CATEGORIES = {
    "color": {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "cyan",
        "magenta",
        "violet",
        "indigo",
        "maroon",
        "navy",
        "olive",
        "lime",
        "aqua",
        "teal",
        "silver",
        "gold",
        "crimson",
        "scarlet",
        "azure",
        "emerald",
        "amber",
        "coral",
        "turquoise",
    },
    "size": {
        "small",
        "large",
        "big",
        "tiny",
        "huge",
        "giant",
        "mini",
        "micro",
        "enormous",
        "massive",
        "compact",
        "medium",
        "oversized",
        "miniature",
        "petite",
        "colossal",
        "vast",
        "minuscule",
        "gigantic",
        "diminutive",
    },
    "shape": {
        "round",
        "square",
        "circular",
        "rectangular",
        "triangular",
        "oval",
        "hexagonal",
        "octagonal",
        "diamond",
        "curved",
        "straight",
        "angular",
        "spherical",
        "cylindrical",
        "conical",
        "flat",
        "pointed",
        "sharp",
    },
    "material": {
        "wooden",
        "metal",
        "plastic",
        "glass",
        "ceramic",
        "fabric",
        "leather",
        "stone",
        "concrete",
        "paper",
        "cardboard",
        "rubber",
        "silicon",
        "steel",
        "aluminum",
        "copper",
        "bronze",
        "gold",
        "silver",
        "wooden",
    },
    "style": {
        "modern",
        "vintage",
        "classic",
        "contemporary",
        "antique",
        "retro",
        "minimalist",
        "elegant",
        "rustic",
        "industrial",
        "artistic",
        "decorative",
        "ornate",
        "simple",
        "complex",
        "traditional",
        "futuristic",
        "baroque",
    },
    "action": {
        "running",
        "flying",
        "sitting",
        "standing",
        "jumping",
        "dancing",
        "walking",
        "swimming",
        "climbing",
        "moving",
        "rotating",
        "spinning",
        "falling",
        "rising",
        "floating",
        "gliding",
        "racing",
        "playing",
    },
    "asset_type": {
        "icon",
        "logo",
        "patch",
        "sticker",
        "image",
        "photo",
        "illustration",
        "shape",
        "symbol",
        "badge",
        "emblem",
        "graphic",
        "design",
        "artwork",
        "picture",
        "drawing",
        "sketch",
        "diagram",
        "chart",
        "banner",
        "vector",
        "bitmap",
        "raster",
        "svg",
        "eps",
        "pdf",
        "template",
        "mockup",
        "prototype",
        "render",
        "clipart",
        "thumbnail",
        "wallpaper",
        "background",
        "texture",
        "pattern",
        "font",
        "typeface",
    },
}


def categorize_attributes(doc):
    """Categorize tokens into different attribute types"""
    categorized = {category: [] for category in ATTRIBUTE_CATEGORIES.keys()}

    for token in doc:
        token_lemma = token.lemma_.lower()
        token_text = token.text.lower()

        # Check both original text and lemma
        for category, values in ATTRIBUTE_CATEGORIES.items():
            if token_lemma in values or token_text in values:
                categorized[category].append(token.text)

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def extract_main_and_sub_objectives(doc):
    """Extract main objective and sub-objectives from the document"""

    # Get noun chunks (these often represent main concepts)
    noun_chunks = list(doc.noun_chunks)

    # Get all nouns and proper nouns
    nouns = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    # Find the core noun (main objective) by looking for the head/root of noun chunks
    main_objective = None
    sub_objectives = []

    # Strategy: Find the root noun that is NOT an attribute
    potential_main_objectives = []

    # First, collect root nouns from noun chunks
    for chunk in noun_chunks:
        root_text = chunk.root.text
        root_lemma = chunk.root.lemma_.lower()

        # Check if this root is an attribute
        is_attribute = False
        for category_values in ATTRIBUTE_CATEGORIES.values():
            if root_lemma in category_values or root_text.lower() in category_values:
                is_attribute = True
                break

        if not is_attribute:
            potential_main_objectives.append(
                {
                    "text": root_text,
                    "chunk": chunk.text,
                    "importance": len(
                        chunk.text.split()
                    ),  # Longer chunks suggest more important context
                }
            )

    # Also check individual nouns that might not be in chunks
    for noun in nouns:
        noun_text = noun.text
        noun_lemma = noun.lemma_.lower()

        # Check if this noun is an attribute
        is_attribute = False
        for category_values in ATTRIBUTE_CATEGORIES.values():
            if noun_lemma in category_values or noun_text.lower() in category_values:
                is_attribute = True
                break

        if not is_attribute:
            # Avoid duplicates
            if not any(
                obj["text"].lower() == noun_text.lower()
                for obj in potential_main_objectives
            ):
                potential_main_objectives.append(
                    {"text": noun_text, "chunk": noun_text, "importance": 1}
                )

    # Sort by importance (nouns from longer chunks are more contextually important)
    potential_main_objectives.sort(key=lambda x: -x["importance"])

    # Set main objective and sub-objectives
    if potential_main_objectives:
        main_objective = potential_main_objectives[0]["text"]

        # Add other non-attribute nouns as sub-objectives
        for obj in potential_main_objectives[1:]:
            if obj["text"].lower() != main_objective.lower():
                sub_objectives.append(obj["text"])

    return main_objective, sub_objectives


def extract_main_intent(query):
    """Enhanced intent extraction with categorization"""
    doc = nlp(query)

    # Extract main and sub objectives
    main_objective, sub_objectives = extract_main_and_sub_objectives(doc)

    # Categorize attributes
    attributes = categorize_attributes(doc)

    # Get all candidate phrases for reference
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    noun_tokens = [token.text for token in doc if token.pos_ == "NOUN"]
    candidates = sorted(set(noun_chunks + noun_tokens), key=lambda x: -len(x))

    return main_objective, sub_objectives, attributes, candidates


def main():
    query = input("Enter your search query: ").strip()

    if not query:
        print("No input provided. Exiting.")
        return

    main_objective, sub_objectives, attributes, candidates = extract_main_intent(query)

    result = {
        "query": query,
        "main_objective": main_objective,
        "sub_objectives": sub_objectives,
        "attributes": attributes,
        "all_candidates": candidates,
    }

    print("\n=== Enhanced Intent Analysis ===")
    print(json.dumps(result, indent=2))

    # Pretty print for better readability
    print("\n=== Structured Output ===")
    print(f"🎯 Main Objective: {main_objective}")
    if sub_objectives:
        print(f"📋 Sub Objectives: {', '.join(sub_objectives)}")

    for attr_type, values in attributes.items():
        if values:
            print(f"🏷️  {attr_type.title()}: {', '.join(values)}")


if __name__ == "__main__":
    main()
