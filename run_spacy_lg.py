import spacy
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache


@dataclass
class NLPConfig:
    """Configuration class for NLP analysis parameters"""

    # Similarity thresholds
    similarity_threshold_format: float = 0.45
    similarity_threshold_artistic: float = 0.6
    similarity_threshold_format_root: float = 0.5

    # Importance scoring
    importance_threshold_sub_objectives: int = 3
    max_sub_objectives: int = 2

    # Base importance scores
    base_importance_score: int = 1
    entity_boost: int = 2
    subject_object_boost: int = 2
    root_boost: int = 1
    adjective_modifier_boost: int = 1
    compound_penalty: int = 1
    noun_chunk_first_boost: int = 3
    noun_chunk_last_boost: int = 1
    generic_penalty: int = 3
    concrete_noun_boost: int = 2
    subject_indicator_boost: int = 2
    format_descriptor_penalty: int = 3

    # POS tags and dependency labels
    past_participle_tags: List[str] = field(default_factory=lambda: ["VBN", "VBD"])
    content_pos_tags: List[str] = field(default_factory=lambda: ["NOUN", "PROPN"])
    modifier_deps: List[str] = field(
        default_factory=lambda: ["amod", "compound", "nmod"]
    )
    subject_object_deps: List[str] = field(
        default_factory=lambda: ["nsubj", "nsubjpass", "dobj", "pobj"]
    )

    # Word lists
    artistic_terms: List[str] = field(
        default_factory=lambda: [
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
    )

    format_concepts: List[str] = field(
        default_factory=lambda: [
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
    )

    artistic_concepts: List[str] = field(
        default_factory=lambda: [
            "drawn",
            "painted",
            "artistic",
            "created",
            "handmade",
            "sketched",
        ]
    )

    false_positive_entities: List[str] = field(
        default_factory=lambda: [
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
    )

    valid_entity_types: List[str] = field(
        default_factory=lambda: [
            "PERSON",
            "GPE",
            "ORG",
            "EVENT",
            "FAC",
            "LOC",
            "PRODUCT",
        ]
    )

    generic_words: List[str] = field(
        default_factory=lambda: [
            "hand",
            "part",
            "piece",
            "thing",
            "item",
            "kit",
            "set",
            "group",
        ]
    )

    compound_parts: List[str] = field(
        default_factory=lambda: [
            "hand",
            "oil",
            "water",
            "digital",
            "computer",
            "machine",
        ]
    )

    artistic_modifiers: List[str] = field(
        default_factory=lambda: ["hand", "oil", "water", "digital", "computer"]
    )

    subject_indicators: List[str] = field(
        default_factory=lambda: [
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
    )

    concrete_nouns: List[str] = field(
        default_factory=lambda: [
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
    )

    colors: List[str] = field(
        default_factory=lambda: [
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
            "beige",
            "tan",
            "burgundy",
            "khaki",
            "lavender",
            "mint",
            "peach",
            "salmon",
            "ivory",
            "cream",
            "charcoal",
        ]
    )


class NLPAnalyzer:
    """Advanced NLP analyzer using spaCy for query intent extraction"""

    def __init__(self, config: Optional[NLPConfig] = None):
        self.config = config or NLPConfig()
        self.nlp = self._load_model()

    @lru_cache(maxsize=1)
    def _load_model(self):
        """Load spaCy model with caching"""
        return spacy.load("en_core_web_lg")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Main method to analyze a query and extract intent"""
        doc = self.nlp(query)

        # Extract main and sub objectives using linguistic analysis
        main_objective, sub_objectives = self._extract_main_objective(doc)

        # Categorize attributes dynamically
        attributes = self._categorize_attributes(doc, main_objective, sub_objectives)

        # Get all candidate phrases for reference
        candidates = self._extract_candidates(doc)

        return {
            "query": query,
            "main_objective": main_objective,
            "sub_objectives": sub_objectives,
            "attributes": attributes,
            "all_candidates": candidates,
        }

    def _extract_main_objective(self, doc) -> Tuple[Optional[str], List[str]]:
        """Extract main objective using pure spaCy linguistic analysis"""
        # Extract compound modifiers first
        compound_modifiers = self._extract_compound_modifiers(doc)

        # Track tokens in compounds to avoid duplicates
        tokens_in_compounds = self._get_tokens_in_compounds(compound_modifiers)

        # Find content candidates
        content_candidates = self._find_content_candidates(doc, tokens_in_compounds)

        # Sort by importance
        content_candidates.sort(key=lambda x: (-x["importance"], x["is_format"]))

        # Set main objective and sub-objectives
        return self._assign_objectives(content_candidates)

    def _extract_compound_modifiers(self, doc) -> List[str]:
        """Extract compound modifiers like 'hand drawn', 'oil painted'"""
        compound_modifiers = []
        doc_tokens = [token for token in doc]

        for i in range(len(doc_tokens) - 1):
            current_token = doc_tokens[i]
            next_token = doc_tokens[i + 1]

            # Check for compound patterns
            if self._is_compound_pattern(current_token, next_token):
                compound_phrase = f"{current_token.text} {next_token.text}"
                if compound_phrase not in compound_modifiers:
                    compound_modifiers.append(compound_phrase)

        return compound_modifiers

    def _is_compound_pattern(self, current_token, next_token) -> bool:
        """Check if two tokens form a compound pattern"""
        # Pattern 1: NOUN + VERB (past participle)
        pattern1 = (
            current_token.pos_ == "NOUN"
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
        )

        # Pattern 2: (NOUN|ADJ) + VERB with specific artistic modifiers
        pattern2 = (
            current_token.pos_ in ["NOUN", "ADJ"]
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
            and current_token.text.lower() in self.config.artistic_modifiers
        )

        return pattern1 or pattern2

    def _get_tokens_in_compounds(self, compound_modifiers: List[str]) -> set:
        """Get set of tokens that are part of compound modifiers"""
        tokens_in_compounds = set()
        for compound in compound_modifiers:
            parts = compound.split()
            for part in parts:
                tokens_in_compounds.add(part.lower())
        return tokens_in_compounds

    def _find_content_candidates(
        self, doc, tokens_in_compounds: set
    ) -> List[Dict[str, Any]]:
        """Find potential content words with importance scoring"""
        content_candidates = []

        for token in doc:
            if token.pos_ in self.config.content_pos_tags:
                # Skip tokens in compound modifiers
                if token.text.lower() in tokens_in_compounds:
                    continue

                importance_score = self._calculate_importance_score(token, doc)
                is_format = self._is_format_descriptor(token)

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

        return content_candidates

    def _calculate_importance_score(self, token, doc) -> int:
        """Calculate importance score for a token"""
        importance_score = self.config.base_importance_score

        # Check if it's a format descriptor
        if self._is_format_descriptor(token):
            importance_score -= self.config.format_descriptor_penalty
        else:
            # Apply various scoring rules
            importance_score += self._apply_scoring_rules(token, doc)

        return importance_score

    def _apply_scoring_rules(self, token, doc) -> int:
        """Apply various scoring rules to determine token importance"""
        score = 0

        # Named entities
        if token.ent_type_ and self._is_valid_entity(token):
            score += self.config.entity_boost

        # Subjects and objects
        if token.dep_ in self.config.subject_object_deps:
            score += self.config.subject_object_boost

        # Root of dependency tree
        if token.dep_ == "ROOT":
            score += self.config.root_boost

        # Tokens with adjective modifiers
        if any(child.dep_ == "amod" for child in token.children):
            score += self.config.adjective_modifier_boost

        # Non-compound modifiers
        if token.dep_ != "compound":
            score += self.config.compound_penalty

        # First in noun chunks
        for chunk in doc.noun_chunks:
            if chunk.start == token.i:
                score += self.config.noun_chunk_first_boost
                break

        # Last in noun chunks
        for chunk in doc.noun_chunks:
            if chunk.end - 1 == token.i:
                score += self.config.noun_chunk_last_boost
                break

        # Generic words penalty
        if token.text.lower() in self.config.generic_words:
            score -= self.config.generic_penalty

        # Subject indicators
        if token.text.lower() in self.config.subject_indicators:
            score += self.config.subject_indicator_boost

        # Concrete nouns
        if token.text.lower() in self.config.concrete_nouns:
            score += self.config.concrete_noun_boost

        # Compound parts penalty
        if token.text.lower() in self.config.compound_parts:
            score -= self.config.compound_penalty

        return score

    def _assign_objectives(
        self, content_candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[str]]:
        """Assign main objective and sub-objectives from candidates"""
        main_objective = None
        sub_objectives = []

        if not content_candidates:
            return main_objective, sub_objectives

        # Find highest-scoring non-format word as main objective
        for candidate in content_candidates:
            if not candidate["is_format"] and candidate["importance"] > 0:
                main_objective = candidate["text"]
                break

        # If no non-format word found, use the highest scoring one
        if not main_objective:
            main_objective = content_candidates[0]["text"]

        # Add other significant content words as sub-objectives
        for candidate in content_candidates:
            if (
                candidate["text"].lower() != main_objective.lower()
                and not candidate["is_format"]
                and candidate["importance"]
                >= self.config.importance_threshold_sub_objectives
                and candidate["dep"] != "compound"
                and not self._is_color(candidate["text"])
            ):
                sub_objectives.append(candidate["text"])
                if len(sub_objectives) >= self.config.max_sub_objectives:
                    break

        return main_objective, sub_objectives

    def _categorize_attributes(
        self, doc, main_objective: Optional[str], sub_objectives: List[str]
    ) -> Dict[str, Any]:
        """Dynamically categorize tokens using spaCy's linguistic analysis"""
        attributes = {
            "adjectives": [],
            "descriptors": [],
            "modifiers": [],
            "actions": [],
            "format_types": [],
            "entities": [],
            "colors": [],
        }

        # Get content words to avoid duplicating in modifiers
        content_words = self._get_content_words(main_objective, sub_objectives)

        # Extract compound modifiers
        compound_modifiers = self._extract_compound_modifiers(doc)
        tokens_in_compounds = self._get_tokens_in_compounds(compound_modifiers)

        # Categorize tokens
        self._categorize_tokens(doc, attributes, content_words, tokens_in_compounds)

        # Add compound modifiers
        if compound_modifiers:
            attributes["modifiers"].extend(compound_modifiers)

        # Post-process attributes
        self._post_process_attributes(doc, attributes, main_objective, sub_objectives)

        return self._finalize_attributes(attributes)

    def _get_content_words(
        self, main_objective: Optional[str], sub_objectives: List[str]
    ) -> set:
        """Get set of main content words"""
        content_words = set()
        if main_objective:
            content_words.add(main_objective.lower())
        if sub_objectives:
            content_words.update([obj.lower() for obj in sub_objectives])
        return content_words

    def _categorize_tokens(
        self,
        doc,
        attributes: Dict[str, List],
        content_words: set,
        tokens_in_compounds: set,
    ):
        """Categorize individual tokens based on their linguistic properties"""
        for token in doc:
            token_text = token.text
            token_lower = token_text.lower()

            # Skip tokens in compounds
            if token_lower in tokens_in_compounds:
                continue

            # Categorize based on POS and other features
            if self._is_color(token):
                attributes["colors"].append(token_text)
            elif (
                token.pos_ in self.config.content_pos_tags
                and self._is_format_descriptor(token)
            ):
                attributes["format_types"].append(token_text)
            elif token.pos_ == "ADJ":
                attributes["adjectives"].append(token_text)
            elif token.pos_ == "ADV":
                attributes["descriptors"].append(token_text)
            elif token.pos_ == "VERB" and token.tag_ == "VBG":
                attributes["actions"].append(token_text)
            elif (
                token.pos_ == "VERB" and token.tag_ in self.config.past_participle_tags
            ):
                if self._is_artistic_descriptor(token):
                    attributes["descriptors"].append(token_text)
                else:
                    attributes["actions"].append(token_text)
            elif token.dep_ in self.config.modifier_deps:
                if (
                    token_text.lower() not in content_words
                    and not self._is_format_descriptor(token)
                    and not self._is_color(token)
                ):
                    attributes["modifiers"].append(token_text)

            # Handle entities
            if token.ent_type_ and self._is_valid_entity(token):
                attributes["entities"].append(
                    {
                        "text": token_text,
                        "type": token.ent_type_,
                        "label": spacy.explain(token.ent_type_),
                    }
                )

    def _post_process_attributes(
        self,
        doc,
        attributes: Dict[str, List],
        main_objective: Optional[str],
        sub_objectives: List[str],
    ):
        """Post-process attributes to add modifiers from noun chunks"""
        if not main_objective:
            return

        for chunk in doc.noun_chunks:
            chunk_tokens = [token.text for token in chunk]
            for token_text in chunk_tokens:
                if self._should_add_as_modifier(
                    token_text, main_objective, sub_objectives, attributes
                ):
                    if "modifiers" not in attributes:
                        attributes["modifiers"] = []
                    attributes["modifiers"].append(token_text)

    def _should_add_as_modifier(
        self,
        token_text: str,
        main_objective: str,
        sub_objectives: List[str],
        attributes: Dict[str, List],
    ) -> bool:
        """Check if token should be added as modifier"""
        token_lower = token_text.lower()

        # Check various exclusion criteria
        if token_lower == main_objective.lower():
            return False
        if token_lower in [obj.lower() for obj in (sub_objectives or [])]:
            return False
        if token_text in attributes.get("format_types", []):
            return False
        if token_text in attributes.get("adjectives", []):
            return False
        if token_text in attributes.get("descriptors", []):
            return False
        if token_text in attributes.get("actions", []):
            return False
        if token_text in attributes.get("colors", []):
            return False
        if self._is_color(token_text):
            return False
        if token_text in [mod.split()[0] for mod in attributes.get("modifiers", [])]:
            return False

        return True

    def _finalize_attributes(self, attributes: Dict[str, List]) -> Dict[str, Any]:
        """Remove empty categories and deduplicate"""
        final_attributes = {}
        for category, items in attributes.items():
            if items:
                if category == "entities":
                    final_attributes[category] = items
                else:
                    final_attributes[category] = list(set(items))
        return final_attributes

    def _is_format_descriptor(self, token) -> bool:
        """Determine if a token is a format/asset type descriptor"""
        # Semantic similarity check
        if token.has_vector:
            for concept in self.config.format_concepts:
                concept_token = self.nlp(concept)[0]
                if concept_token.has_vector:
                    similarity = token.similarity(concept_token)
                    if similarity > self.config.similarity_threshold_format:
                        return True

        # Suffix pattern check
        token_lower = token.text.lower()
        if (
            token_lower.endswith(("ic", "ics"))
            and token.pos_ in self.config.content_pos_tags
        ):
            return True

        # Root with compound children check
        if token.dep_ == "ROOT" and token.pos_ in self.config.content_pos_tags:
            compound_children = [
                child for child in token.children if child.dep_ == "compound"
            ]
            if len(compound_children) >= 2 and token.has_vector:
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
                    concept_token = self.nlp(concept)[0]
                    if concept_token.has_vector:
                        similarity = token.similarity(concept_token)
                        if similarity > self.config.similarity_threshold_format_root:
                            return True

        return False

    def _is_artistic_descriptor(self, token) -> bool:
        """Check if a past participle is an artistic/style descriptor"""
        token_lower = token.text.lower()

        # Direct term match
        if token_lower in self.config.artistic_terms:
            return True

        # Semantic similarity check
        if token.has_vector:
            for concept in self.config.artistic_concepts:
                concept_token = self.nlp(concept)[0]
                if concept_token.has_vector:
                    similarity = token.similarity(concept_token)
                    if similarity > self.config.similarity_threshold_artistic:
                        return True

        return False

    def _is_valid_entity(self, token) -> bool:
        """Filter out obvious false positive entities"""
        token_lower = token.text.lower()

        # Check against false positives
        if token_lower in self.config.false_positive_entities:
            return False

        # Check valid entity types
        return token.ent_type_ in self.config.valid_entity_types

    def _is_color(self, token_or_text) -> bool:
        """Check if a token or text is a color"""
        if isinstance(token_or_text, str):
            return token_or_text.lower() in self.config.colors
        else:
            return token_or_text.text.lower() in self.config.colors

    def _extract_candidates(self, doc) -> List[str]:
        """Extract all candidate phrases for reference"""
        noun_chunks = []
        noun_tokens = []

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if chunk.text.strip():
                noun_chunks.append(chunk.text.strip())

        # Extract individual noun tokens
        for token in doc:
            if token.pos_ in self.config.content_pos_tags and token.text.strip():
                noun_tokens.append(token.text.strip())

        # Combine and deduplicate
        all_candidates = noun_chunks + noun_tokens
        return sorted(set(all_candidates), key=lambda x: -len(x))


def main():
    """Main function to run the NLP analyzer"""
    query = input("Enter your search query: ").strip()

    if not query:
        print("No input provided. Exiting.")
        return

    # Initialize analyzer
    analyzer = NLPAnalyzer()

    # Analyze query
    result = analyzer.analyze_query(query)

    # Display results
    print("\n=== Pure spaCy Linguistic Analysis ===")
    print(json.dumps(result, indent=2))

    # Pretty print for better readability
    print("\n=== Structured Output ===")
    print(f"🎯 Main Objective: {result['main_objective']}")
    if result["sub_objectives"]:
        print(f"📋 Sub Objectives: {', '.join(result['sub_objectives'])}")

    for attr_type, values in result["attributes"].items():
        if values:
            if attr_type == "entities":
                entity_strs = [f"{item['text']} ({item['label']})" for item in values]
                print(f"🏷️  {attr_type.title()}: {', '.join(entity_strs)}")
            elif attr_type == "format_types":
                print(f"📄 Format Types: {', '.join(values)}")
            elif attr_type == "colors":
                print(f"🎨 Colors: {', '.join(values)}")
            else:
                print(f"🏷️  {attr_type.title().replace('_', ' ')}: {', '.join(values)}")


if __name__ == "__main__":
    main()
