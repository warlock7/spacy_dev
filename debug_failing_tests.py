#!/usr/bin/env python3
"""Debug script for failing test cases"""

import spacy
from nlp_analyzer_architecture import NLPAnalyzer


def debug_query(query):
    print(f"\n{'='*60}")
    print(f"DEBUGGING: '{query}'")
    print(f"{'='*60}")

    # Check spaCy parsing
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(query)

    print("SpaCy Analysis:")
    for token in doc:
        print(
            f"  {token.text:<12} | POS: {token.pos_:<6} | DEP: {token.dep_:<12} | HEAD: {token.head.text}"
        )

    print("\nNoun chunks:")
    for chunk in doc.noun_chunks:
        print(f"  '{chunk.text}' -> root: {chunk.root.text}")

    # Check analyzer result
    analyzer = NLPAnalyzer()

    # Debug the actual assignment process
    print(f"\nDEBUGGING ASSIGNMENT PROCESS:")
    obj_extractor = analyzer.objective_extractor

    # Get the same content candidates the algorithm uses
    compound_extractor = obj_extractor._get_tokens_in_compounds([])
    content_candidates = obj_extractor._find_content_candidates(doc, set())

    print(f"Raw content candidates from _find_content_candidates:")
    for candidate in content_candidates:
        print(
            f"  {candidate['text']:<12} | Score: {candidate['importance']:<3} | Format: {candidate['is_format']} | Attr: {candidate['is_attribute']}"
        )

    # Sort like the algorithm does
    content_candidates.sort(key=lambda x: -x["importance"])
    print(f"\nAfter sorting by importance:")
    for candidate in content_candidates:
        print(
            f"  {candidate['text']:<12} | Score: {candidate['importance']:<3} | Format: {candidate['is_format']} | Attr: {candidate['is_attribute']}"
        )

    # Trace assignment logic
    main_objective = None

    # Priority 1: non-format, non-attribute, positive score
    print(f"\nPriority 1 check (non-format, non-attribute, positive score):")
    for candidate in content_candidates:
        print(
            f"  {candidate['text']}: score={candidate['importance']}>0? {candidate['importance']>0}, not_attr? {not analyzer.attribute_manager.is_attribute(candidate['text'])}, not_format? {not candidate['is_format']}"
        )
        if (
            candidate["importance"] > 0
            and not analyzer.attribute_manager.is_attribute(candidate["text"])
            and not candidate["is_format"]
        ):
            main_objective = candidate["text"]
            print(f"    -> SELECTED: {main_objective}")
            break

    if not main_objective:
        print(f"  -> No Priority 1 match")

        # Priority 2: non-attribute, positive score (allows format)
        print(f"\nPriority 2 check (non-attribute, positive score, allows format):")
        for candidate in content_candidates:
            print(
                f"  {candidate['text']}: score={candidate['importance']}>0? {candidate['importance']>0}, not_attr? {not analyzer.attribute_manager.is_attribute(candidate['text'])}"
            )
            if candidate[
                "importance"
            ] > 0 and not analyzer.attribute_manager.is_attribute(candidate["text"]):
                main_objective = candidate["text"]
                print(f"    -> SELECTED: {main_objective}")
                break

        if not main_objective:
            print(f"  -> No Priority 2 match")
            print(f"\nPriority 3 check (any non-attribute):")
            for candidate in content_candidates:
                if not analyzer.attribute_manager.is_attribute(candidate["text"]):
                    main_objective = candidate["text"]
                    print(f"    -> SELECTED: {main_objective}")
                    break

    print(f"\nFINAL ASSIGNMENT: {main_objective}")

    # Compare with actual result
    result = analyzer.analyze_query(query)
    print(f"ACTUAL RESULT: {result['main_objective']}")
    print(f"MATCH: {main_objective == result['main_objective']}")


if __name__ == "__main__":
    debug_query("apple sketch")
    debug_query("large oil painting by Pablo Picasso")
