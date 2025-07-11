import spacy
import json

# 📌 Load spaCy large model
nlp = spacy.load("en_core_web_lg")


def categorize_attributes_dynamically(doc, main_objective=None, sub_objectives=None):
    """Dynamically categorize tokens using spaCy's linguistic analysis only"""
    attributes = {
        "adjectives": [],
        "descriptors": [],
        "modifiers": [],
        "actions": [],
        "format_types": [],  # NEW: For capturing format descriptors
        "entities": [],
    }

    # Get list of main content words to avoid duplicating in modifiers
    content_words = set()
    if main_objective:
        content_words.add(main_objective.lower())
    if sub_objectives:
        content_words.update([obj.lower() for obj in sub_objectives])

    # First pass: identify compound modifiers like "hand drawn"
    compound_modifiers = []
    doc_tokens = [token for token in doc]

    for i in range(len(doc_tokens) - 1):
        current_token = doc_tokens[i]
        next_token = doc_tokens[i + 1]

        # Look for patterns like "hand drawn", "oil painted", etc.
        if (
            current_token.pos_ == "NOUN"
            and next_token.pos_ == "VERB"
            and next_token.tag_
            in ["VBN", "VBD"]  # Handle both past participle and past tense
            and is_artistic_descriptor(next_token)
        ):

            compound_phrase = f"{current_token.text} {next_token.text}"
            compound_modifiers.append(compound_phrase)

    # Also check for adjective + past participle patterns like "hand painted"
    for i in range(len(doc_tokens) - 1):
        current_token = doc_tokens[i]
        next_token = doc_tokens[i + 1]

        # Look for patterns like "hand painted" where "hand" might be tagged as ADJ
        if (
            current_token.pos_ in ["NOUN", "ADJ"]
            and next_token.pos_ == "VERB"
            and next_token.tag_
            in ["VBN", "VBD"]  # Handle both past participle and past tense
            and is_artistic_descriptor(next_token)
            and current_token.text.lower()
            in ["hand", "oil", "water", "digital", "computer"]
        ):

            compound_phrase = f"{current_token.text} {next_token.text}"
            # Avoid duplicates
            if compound_phrase not in compound_modifiers:
                compound_modifiers.append(compound_phrase)

    # Track tokens that are part of compound modifiers
    tokens_in_compounds = set()
    for compound in compound_modifiers:
        parts = compound.split()
        for part in parts:
            tokens_in_compounds.add(part.lower())

    for token in doc:
        token_text = token.text
        token_lower = token_text.lower()

        # Skip tokens that are part of compound modifiers
        if token_lower in tokens_in_compounds:
            continue

        # 1. Check if this is a format descriptor first
        if token.pos_ in ["NOUN", "PROPN"] and is_format_descriptor(token):
            attributes["format_types"].append(token_text)

        # 2. Adjectives are natural descriptive attributes
        elif token.pos_ == "ADJ":
            attributes["adjectives"].append(token_text)

        # 3. Adverbs that modify adjectives/verbs (descriptors)
        elif token.pos_ == "ADV":
            attributes["descriptors"].append(token_text)

        # 4. Verbs in present participle form (actions like "flying", "running")
        elif token.pos_ == "VERB" and token.tag_ == "VBG":
            attributes["actions"].append(token_text)

        # 5. Past participles used as descriptors (drawn, painted, carved, etc.)
        elif token.pos_ == "VERB" and token.tag_ in [
            "VBN",
            "VBD",
        ]:  # Handle both past participle and past tense
            # Check if it's an artistic/style descriptor
            if is_artistic_descriptor(token):
                attributes["descriptors"].append(token_text)
            else:
                attributes["actions"].append(token_text)

        # 6. Dependency-based modifiers - but exclude main content words and format types
        elif token.dep_ in ["amod", "compound", "nmod"]:
            # Only add as modifier if it's not the main objective, sub-objective, or format type
            if token_text.lower() not in content_words and not is_format_descriptor(
                token
            ):
                attributes["modifiers"].append(token_text)

        # 7. Named entities (people, places, organizations, etc.) - but filter out obvious false positives
        if token.ent_type_ and is_valid_entity(token):
            attributes["entities"].append(
                {
                    "text": token_text,
                    "type": token.ent_type_,
                    "label": spacy.explain(token.ent_type_),
                }
            )

    # Add compound modifiers
    if compound_modifiers:
        if "modifiers" not in attributes:
            attributes["modifiers"] = []
        attributes["modifiers"].extend(compound_modifiers)

    # Post-processing: Add non-objective words from noun chunks as modifiers
    if main_objective:
        for chunk in doc.noun_chunks:
            chunk_tokens = [token.text for token in chunk]
            for token_text in chunk_tokens:
                if (
                    token_text.lower() != main_objective.lower()
                    and token_text.lower()
                    not in [obj.lower() for obj in (sub_objectives or [])]
                    and token_text not in attributes.get("format_types", [])
                    and token_text not in attributes.get("adjectives", [])
                    and token_text not in attributes.get("descriptors", [])
                    and token_text not in attributes.get("actions", [])
                    and token_text
                    not in [mod.split()[0] for mod in attributes.get("modifiers", [])]
                ):  # Avoid duplicates from compound modifiers

                    if "modifiers" not in attributes:
                        attributes["modifiers"] = []
                    attributes["modifiers"].append(token_text)

    # Remove empty categories and deduplicate
    final_attributes = {}
    for category, items in attributes.items():
        if items:
            if category == "entities":
                final_attributes[category] = items
            else:
                final_attributes[category] = list(set(items))

    return final_attributes


def is_format_descriptor(token):
    """Determine if a token is a format/asset type descriptor using linguistic analysis"""

    # 1. Check semantic context using word vectors - primary method
    if token.has_vector:
        # Core format/technical concepts for semantic similarity
        format_concepts = [
            "image",
            "graphic",
            "design",
            "artwork",
            "illustration",
            "logo",
            "icon",
            "symbol",
            "vector",
            "format",
            "file",
            "style",
        ]

        for concept in format_concepts:
            concept_token = nlp(concept)[0]
            if concept_token.has_vector:
                similarity = token.similarity(concept_token)
                if (
                    similarity > 0.45
                ):  # Lowered threshold to catch style descriptors like silhouette
                    return True

    # 2. Check for typical format suffixes or patterns
    token_lower = token.text.lower()
    if token_lower.endswith(("ic", "ics")) and token.pos_ in ["NOUN", "PROPN"]:
        # Words ending in these suffixes often describe format/type
        return True

    # 3. Position-based logic: if ROOT has multiple compound children, check if it's format
    if token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]:
        compound_children = [
            child for child in token.children if child.dep_ == "compound"
        ]
        if len(compound_children) >= 2:
            # If root has multiple compound modifiers, it might be format descriptor
            # But only if it has high semantic similarity to format concepts
            if token.has_vector:
                for concept in [
                    "image",
                    "graphic",
                    "design",
                    "artwork",
                    "illustration",
                    "logo",
                    "icon",
                    "vector",
                ]:
                    concept_token = nlp(concept)[0]
                    if concept_token.has_vector:
                        similarity = token.similarity(concept_token)
                        if similarity > 0.5:  # Lower threshold for this specific case
                            return True

    return False


def is_artistic_descriptor(token):
    """Check if a past participle is an artistic/style descriptor"""
    token_lower = token.text.lower()

    # Common artistic descriptors
    artistic_terms = [
        "drawn",
        "painted",
        "carved",
        "sculpted",
        "sketched",
        "illustrated",
        "rendered",
        "crafted",
        "designed",
        "created",
        "handmade",
        "traced",
        "etched",
        "printed",
        "digitized",
        "stylized",
        "engraved",
        "embossed",
    ]

    if token_lower in artistic_terms:
        return True

    # Check semantic similarity to artistic concepts
    if token.has_vector:
        artistic_concepts = [
            "drawn",
            "painted",
            "artistic",
            "created",
            "handmade",
            "sketched",
        ]
        for concept in artistic_concepts:
            concept_token = nlp(concept)[0]
            if concept_token.has_vector:
                similarity = token.similarity(concept_token)
                if similarity > 0.6:  # High threshold for artistic descriptors
                    return True

    return False


def is_valid_entity(token):
    """Filter out obvious false positive entities"""
    token_lower = token.text.lower()

    # Common words that get incorrectly tagged as entities
    false_positives = [
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "orange",
        "purple",
        "apple",
        "cat",
        "dog",
        "flower",
        "tree",
        "house",
        "car",
        "book",
        "small",
        "large",
        "big",
        "little",
        "old",
        "new",
        "good",
        "bad",
        "digital",
        "analog",
        "manual",
        "automatic",
        "modern",
        "classic",
        "wooden",
        "metal",
        "plastic",
        "glass",
        "stone",
        "paper",
    ]

    # Don't include common adjectives/nouns as entities
    if token_lower in false_positives:
        return False

    # Only include entities that are likely to be actual named entities
    valid_entity_types = ["PERSON", "GPE", "ORG", "EVENT", "FAC", "LOC", "PRODUCT"]
    return token.ent_type_ in valid_entity_types


def extract_main_objective_linguistically(doc):
    """Extract main objective using pure spaCy linguistic analysis"""

    main_objective = None
    sub_objectives = []

    # First, identify compound modifiers like "hand drawn"
    compound_modifiers = []
    doc_tokens = [token for token in doc]

    for i in range(len(doc_tokens) - 1):
        current_token = doc_tokens[i]
        next_token = doc_tokens[i + 1]

        # Look for patterns like "hand drawn", "oil painted", etc.
        if (
            current_token.pos_ == "NOUN"
            and next_token.pos_ == "VERB"
            and next_token.tag_
            in ["VBN", "VBD"]  # Handle both past participle and past tense
            and is_artistic_descriptor(next_token)
        ):

            compound_phrase = f"{current_token.text} {next_token.text}"
            compound_modifiers.append(compound_phrase)

    # Also check for adjective + past participle patterns like "hand painted"
    for i in range(len(doc_tokens) - 1):
        current_token = doc_tokens[i]
        next_token = doc_tokens[i + 1]

        # Look for patterns like "hand painted" where "hand" might be tagged as ADJ
        if (
            current_token.pos_ in ["NOUN", "ADJ"]
            and next_token.pos_ == "VERB"
            and next_token.tag_
            in ["VBN", "VBD"]  # Handle both past participle and past tense
            and is_artistic_descriptor(next_token)
            and current_token.text.lower()
            in ["hand", "oil", "water", "digital", "computer"]
        ):

            compound_phrase = f"{current_token.text} {next_token.text}"
            # Avoid duplicates
            if compound_phrase not in compound_modifiers:
                compound_modifiers.append(compound_phrase)

    # Track tokens that are part of compound modifiers - these should NOT be objectives
    tokens_in_compounds = set()
    for compound in compound_modifiers:
        parts = compound.split()
        for part in parts:
            tokens_in_compounds.add(part.lower())

    # Find all potential content words (nouns and proper nouns)
    content_candidates = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:

            # Skip tokens that are part of compound modifiers
            if token.text.lower() in tokens_in_compounds:
                continue

            # Calculate semantic importance score
            importance_score = 1  # Base score

            # Check if this is likely a format descriptor
            is_format = is_format_descriptor(token)

            if is_format:
                importance_score -= 3  # Heavy penalty for format descriptors
            else:
                # Boost score for content indicators

                # 1. Named entities are usually important content (if valid)
                if token.ent_type_ and is_valid_entity(token):
                    importance_score += 2

                # 2. Subjects and objects are typically main content
                if token.dep_ in ["nsubj", "nsubjpass", "dobj", "pobj"]:
                    importance_score += 2

                # 3. Root of dependency tree is important, but not always the main content
                if token.dep_ == "ROOT":
                    importance_score += 1  # Reduced from 3 to 1

                # 4. Tokens with adjective modifiers are often main concepts
                if any(child.dep_ == "amod" for child in token.children):
                    importance_score += 1

                # 5. Tokens that are NOT compound modifiers are more likely content
                if token.dep_ != "compound":
                    importance_score += 1

                # 6. Boost score for tokens that appear first in noun chunks
                # This helps prioritize "apple" in "apple hand drawn" and "football" in "football kit"
                for chunk in doc.noun_chunks:
                    if chunk.start == token.i:
                        importance_score += 3  # Increased from 2 to 3
                        break

                # 7. Penalty for tokens that are too generic or common
                if token.text.lower() in [
                    "hand",
                    "part",
                    "piece",
                    "thing",
                    "item",
                    "kit",
                    "set",
                    "group",
                ]:
                    importance_score -= (
                        3  # Increased penalty from 2 to 3 to prevent sub-objectives
                    )

                # 8. Boost score for tokens that appear at the end of phrases
                # This helps prioritize "portrait" in "digital hand sketched portrait"
                for chunk in doc.noun_chunks:
                    if chunk.end - 1 == token.i:  # Last token in noun chunk
                        importance_score += 1
                        break

                # 9. Boost score for words that are likely to be main subjects
                subject_indicators = [
                    "portrait",
                    "painting",
                    "drawing",
                    "sketch",
                    "illustration",
                    "photo",
                    "image",
                    "picture",
                    "artwork",
                    "design",
                    "logo",
                    "icon",
                    "symbol",
                    "animal",
                    "person",
                    "face",
                    "landscape",
                    "building",
                    "car",
                    "tree",
                    "flower",
                    "bird",
                    "cat",
                    "dog",
                    "horse",
                    "elephant",
                    "lion",
                    "border",
                    "frame",
                    "pattern",
                    "background",
                    "texture",
                    "banner",
                ]
                if token.text.lower() in subject_indicators:
                    importance_score += 2

                # 10. Extra boost for concrete nouns vs abstract/generic ones
                concrete_nouns = [
                    "apple",
                    "football",
                    "soccer",
                    "basketball",
                    "tennis",
                    "baseball",
                    "car",
                    "house",
                    "tree",
                    "flower",
                    "cat",
                    "dog",
                    "bird",
                    "fish",
                    "border",
                    "frame",
                    "pattern",
                    "background",
                    "texture",
                    "banner",
                ]
                if token.text.lower() in concrete_nouns:
                    importance_score += 2

                # 11. NEW: Extra penalty for words that are commonly part of compound modifiers
                compound_parts = [
                    "hand",
                    "oil",
                    "water",
                    "digital",
                    "computer",
                    "machine",
                ]
                if token.text.lower() in compound_parts:
                    importance_score -= 1  # Additional penalty to prevent these from being main objectives

            content_candidates.append(
                {
                    "text": token.text,
                    "importance": importance_score,
                    "is_format": is_format,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "ent_type": token.ent_type_,
                }
            )

    # Sort by importance (content words first, then by score)
    content_candidates.sort(key=lambda x: (-x["importance"], x["is_format"]))

    # Set main objective and sub-objectives
    if content_candidates:
        # Find the highest-scoring non-format word as main objective
        for candidate in content_candidates:
            if not candidate["is_format"] and candidate["importance"] > 0:
                main_objective = candidate["text"]
                break

        # If no non-format word found, use the highest scoring one
        if not main_objective and content_candidates:
            main_objective = content_candidates[0]["text"]

        # Add other significant content words as sub-objectives
        # Look for secondary content words (nouns that aren't format descriptors)
        for candidate in content_candidates:
            if (
                candidate["text"].lower() != main_objective.lower()
                and not candidate["is_format"]
                and candidate["importance"] >= 3  # Higher threshold for sub-objectives
                and candidate["dep"]
                != "compound"  # Exclude compound modifiers from sub-objectives
            ):
                sub_objectives.append(candidate["text"])
                # Limit sub-objectives to avoid clutter
                if len(sub_objectives) >= 2:
                    break

    return main_objective, sub_objectives


def extract_main_intent(query):
    """Pure spaCy-based intent extraction without hardcoded lists"""
    doc = nlp(query)

    # Extract main and sub objectives using linguistic analysis
    main_objective, sub_objectives = extract_main_objective_linguistically(doc)

    # Categorize attributes dynamically - pass main objective and sub objectives to avoid duplication
    attributes = categorize_attributes_dynamically(doc, main_objective, sub_objectives)

    # Get all candidate phrases for reference - improved extraction
    noun_chunks = []
    noun_tokens = []

    # Extract noun chunks more reliably
    for chunk in doc.noun_chunks:
        if chunk.text.strip():  # Ensure non-empty chunks
            noun_chunks.append(chunk.text.strip())

    # Extract individual noun tokens
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text.strip():
            noun_tokens.append(token.text.strip())

    # Combine and deduplicate candidates
    all_candidates = noun_chunks + noun_tokens
    candidates = sorted(set(all_candidates), key=lambda x: -len(x))

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

    print("\n=== Pure spaCy Linguistic Analysis ===")
    print(json.dumps(result, indent=2))

    # Pretty print for better readability
    print("\n=== Structured Output ===")
    print(f"🎯 Main Objective: {main_objective}")
    if sub_objectives:
        print(f"📋 Sub Objectives: {', '.join(sub_objectives)}")

    for attr_type, values in attributes.items():
        if values:
            if attr_type == "entities":
                entity_strs = [f"{item['text']} ({item['label']})" for item in values]
                print(f"🏷️  {attr_type.title()}: {', '.join(entity_strs)}")
            elif attr_type == "format_types":
                print(f"📄 Format Types: {', '.join(values)}")
            else:
                print(f"🏷️  {attr_type.title().replace('_', ' ')}: {', '.join(values)}")


if __name__ == "__main__":
    main()
