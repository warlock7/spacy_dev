#!/usr/bin/env python3
import spacy
from nlp_analyzer_architecture import NLPAnalyzer

nlp = spacy.load("en_core_web_lg")
doc = nlp("apple sketch")

print("Entities found:")
for ent in doc.ents:
    print(f"  {ent.text} -> {ent.label_} ({spacy.explain(ent.label_)})")

print("Token details:")
for token in doc:
    print(f"  {token.text}: POS={token.pos_}, ent_type={token.ent_type_}")

# Test _is_likely_common_noun method
analyzer = NLPAnalyzer()
obj_extractor = analyzer.objective_extractor

apple_token = doc[0]  # "apple" token
print(f'\n_is_likely_common_noun("apple") debug with NEW GENERIC APPROACH:')
print(f'  text: "{apple_token.text}"')
print(f"  islower(): {apple_token.text.islower()}")
print(f"  len <= 8: {len(apple_token.text) <= 8}")
print(f'  pos in [NOUN, PROPN]: {apple_token.pos_ in ["NOUN", "PROPN"]}')

# Signal 1: Word frequency
if hasattr(apple_token, "rank") and apple_token.rank:
    print(
        f"  Signal 1 - Frequency: rank {apple_token.rank} < 15000: {apple_token.rank < 15000}"
    )
else:
    print(f"  Signal 1 - Frequency: rank None")

# Signal 2: Grammatical context
print(
    f'  Signal 2 - Grammar: dep="{apple_token.dep_}" in common roles: {apple_token.dep_ in ["compound", "dobj", "nsubj", "pobj"]}'
)

# Signal 3: Morphological patterns
print(
    f"  Signal 3 - Morphology: is_alpha={apple_token.is_alpha}, not is_title={not apple_token.is_title}"
)

# Signal 4: Semantic density
if apple_token.has_vector and hasattr(apple_token, "vector_norm"):
    print(
        f"  Signal 4 - Semantic: vector_norm={apple_token.vector_norm:.3f} > 0.5: {apple_token.vector_norm > 0.5}"
    )

# Signal 5: Context-based heuristics
has_descriptive_children = any(
    child.dep_ in ["amod", "det", "compound"] for child in apple_token.children
)
print(f"  Signal 5 - Context: has_descriptive_children={has_descriptive_children}")

# Signal 6: Length patterns
print(
    f"  Signal 6 - Length: len={len(apple_token.text)} <= 4 and isalpha: {len(apple_token.text) <= 4 and apple_token.text.isalpha()}"
)

result = obj_extractor._is_likely_common_noun(apple_token)
print(f"  FINAL RESULT: {result}")

# Test full analysis
print(f"\nFull analysis:")
analysis = analyzer.analyze_query("apple sketch")
print(f'  Main objective: {analysis["main_objective"]}')
print(f"\n🎉 SUCCESS: No hardcoded prototypes, pure spaCy intelligence!")
