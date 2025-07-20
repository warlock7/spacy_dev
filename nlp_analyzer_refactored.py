import json
import numpy as np
import webcolors
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Sequence

import spacy
from spacy.tokens import Doc, Token

from config import NLPConfig
from layers import AttributeManager, SemanticAttributeDetector, ObjectiveClassifier

# -----------------------------------------------------------------------------
# Semantic attribute detectors (top-level)
# -----------------------------------------------------------------------------


class NLPAnalyzer:
    """Advanced NLP analyzer using spaCy for query intent extraction.

    This implementation is self-contained and depends only on `config.py` for
    configuration parameters, allowing easy experimentation without touching the
    original `run_spacy_lg.py`.  Functionality is (for now) identical to the
    heuristic-based analyzer you already used, but prepared for incremental
    refactors (e.g. multi-layer classification, caching prototype vectors, etc.).
    """

    def __init__(self, config: Optional[NLPConfig] = None):
        self.config = config or NLPConfig()
        self.nlp = self._load_model()
        # Pre-compute prototype tokens once for fast similarity checks
        self._format_proto_tokens = [
            self.nlp(concept)[0] for concept in self.config.format_concepts
        ]
        self._artistic_proto_tokens = [
            self.nlp(concept)[0] for concept in self.config.artistic_concepts
        ]
        self._size_proto_tokens = [
            self.nlp(prototype)[0] for prototype in self.config.size_prototypes
        ]
        self._material_proto_tokens = [
            self.nlp(material)[0] for material in self.config.material_prototypes
        ]
        self._shape_proto_tokens = [
            self.nlp(shape)[0] for shape in self.config.shape_prototypes
        ]

        # Concrete vs abstract centroids for concreteness scoring
        self._concrete_centroid = self._compute_centroid(
            self.config.concrete_prototypes
        )
        self._abstract_centroid = self._compute_centroid(
            self.config.abstract_prototypes
        )

        # Attribute manager owning semantic detectors and future layers
        self.attribute_manager = AttributeManager(self)
        self.objective_classifier = ObjectiveClassifier(self)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Main entry point: returns a structured intent breakdown."""
        doc = self.nlp(query)

        main_obj, sub_objs = self._determine_objectives(doc)
        attrs = self._categorize_attributes(doc, main_obj, sub_objs)
        candidates = self._extract_candidates(doc)

        return {
            "query": query,
            "main_objective": main_obj,
            "sub_objectives": sub_objs,
            "attributes": attrs,
            "all_candidates": candidates,
        }

    # ------------------------------------------------------------------
    # Model loading / caching helpers
    # ------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def _load_model(self):
        """Lazy-load the large English spaCy model once."""
        try:
            return spacy.load("en_core_web_lg")
        except OSError as err:
            raise RuntimeError(
                "spaCy model 'en_core_web_lg' is not installed. "
                "Run: python -m spacy download en_core_web_lg"
            ) from err

    # ------------------------------------------------------------------
    # Objective extraction
    # ------------------------------------------------------------------
    def _determine_objectives(self, doc: Doc) -> Tuple[Optional[str], List[str]]:
        compound_mods = self._extract_compound_modifiers(doc)
        tokens_in_comp = self._get_tokens_in_compounds(compound_mods)
        candidates = self._find_content_candidates(doc, tokens_in_comp)
        candidates.sort(key=lambda x: (-x["importance"], x["is_format"]))
        print("candidates =====> ", candidates)
        return self.objective_classifier.choose(candidates)

    # Re-implement helper methods verbatim from the original analyzer --------

    # ---------------- Compound modifiers ----------------
    def _extract_compound_modifiers(self, doc: Doc) -> List[str]:
        """Return phrases like 'hand drawn' that behave as compound modifiers."""
        compound_phrases: List[str] = []
        for i in range(len(doc) - 1):
            current_token, next_token = doc[i], doc[i + 1]
            if self._is_compound_pattern(current_token, next_token):
                phrase = f"{current_token.text} {next_token.text}"
                if phrase not in compound_phrases:
                    compound_phrases.append(phrase)
        return compound_phrases

    def _is_compound_pattern(self, current_token: Token, next_token: Token) -> bool:
        """Detect (token_i, token_{i+1}) patterns that form artistic compounds."""
        pattern1 = (
            current_token.pos_ == "NOUN"
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
        )
        pattern2 = (
            current_token.pos_ in ["NOUN", "ADJ"]
            and next_token.pos_ == "VERB"
            and next_token.tag_ in self.config.past_participle_tags
            and self._is_artistic_descriptor(next_token)
        )
        return pattern1 or pattern2

    def _get_tokens_in_compounds(self, compounds: List[str]) -> set[str]:
        tokens = set()
        for compound in compounds:
            tokens.update([part.lower() for part in compound.split()])
        return tokens

    # ---------------- Candidate discovery ----------------
    def _find_content_candidates(
        self, doc: Doc, tokens_in_comp: set[str]
    ) -> List[Dict[str, Any]]:
        """Collect noun/proper-noun tokens that could be objectives."""
        candidates: List[Dict[str, Any]] = []
        for token in doc:
            if (
                token.pos_ in self.config.content_pos_tags
                and token.text.lower() not in tokens_in_comp
            ):
                importance = self._calculate_importance_score(token, doc)
                candidate = {
                    "text": token.text,
                    "importance": importance,
                    "is_format": self._is_format_descriptor(token),
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "ent_type": token.ent_type_,
                    "token": token,
                }
                candidates.append(candidate)
        return candidates

    # Importance calculation -------------------------------------------------
    def _calculate_importance_score(self, token: Token, doc: Doc) -> int:
        """Return heuristic importance score for a single token."""
        score = self.config.base_importance_score
        if self._is_format_descriptor(token):
            score -= self.config.format_descriptor_penalty
        else:
            score += self._apply_scoring_rules(token, doc)
        return score

    def _apply_scoring_rules(self, token: Token, doc: Doc) -> int:
        """Detailed scoring heuristics for a content token."""
        score = 0

        # Named-entity relevance
        if token.ent_type_ and self._is_valid_entity(token):
            score += self.config.entity_boost

        # Grammatical role boosts
        if token.dep_ in self.config.subject_object_deps:
            score += self.config.subject_object_boost
        if token.dep_ == "ROOT":
            score += self.config.root_boost

        # Adjective modifiers indicate descriptive importance
        if any(child.dep_ == "amod" for child in token.children):
            score += self.config.adjective_modifier_boost

        # Penalise compound parts (token acts as modifier, not head)
        if token.dep_ != "compound":
            score += self.config.compound_penalty

        # Position inside noun chunk
        for chunk in doc.noun_chunks:
            if chunk.start == token.i:
                score += self.config.noun_chunk_first_boost
            if chunk.end - 1 == token.i:
                score += self.config.noun_chunk_last_boost

        # Word-list penalties / boosts
        token_lower = token.text.lower()
        # Frequency-based generic-word penalty (replaces long word list)
        if (
            token.pos_ in {"NOUN", "PROPN"}
            and self._get_lexeme_rank(token) < self.config.generic_word_rank_threshold
        ):
            score -= self.config.generic_penalty
        # Boost tokens that are grammatical subjects/objects AND concrete
        if token.dep_ in self.config.subject_object_deps and self._is_concrete(token):
            score += self.config.subject_indicator_boost
        # Concreteness boost based on centroid similarity
        if self._is_concrete(token):
            score += self.config.concrete_noun_boost

        return score

    # ------------------------------------------------------------------
    # Attribute categorization (same logic as before)
    # ------------------------------------------------------------------
    def _categorize_attributes(
        self, doc: Doc, main_obj: Optional[str], sub_objs: List[str]
    ) -> Dict[str, Any]:
        return self.attribute_manager.categorize_attributes(doc, main_obj, sub_objs)

    # Helper functions _post_process_attributes and _finalize_attributes are moved into AttributeManager; retain for external references if not used.

    # ------------------------------------------------------------------
    # Similarity / descriptor helpers
    # ------------------------------------------------------------------
    def _is_format_descriptor(self, token: Token) -> bool:
        if token.has_vector:
            for proto in self._format_proto_tokens:
                if (
                    proto.has_vector
                    and token.similarity(proto)
                    > self.config.similarity_threshold_format
                ):
                    return True
        token_lower = token.text.lower()
        if (
            token_lower.endswith(("ic", "ics"))
            and token.pos_ in self.config.content_pos_tags
        ):
            return True
        if token.dep_ == "ROOT" and token.pos_ in self.config.content_pos_tags:
            compound_children = [
                child for child in token.children if child.dep_ == "compound"
            ]
            if len(compound_children) >= 2 and token.has_vector:
                for proto in self._format_proto_tokens:
                    if (
                        token.similarity(proto)
                        > self.config.similarity_threshold_format_root
                    ):
                        return True
        return False

    def _is_artistic_descriptor(self, token: Token) -> bool:
        if token.text.lower() in self.config.artistic_terms:
            return True
        if token.has_vector:
            for proto in self._artistic_proto_tokens:
                if token.similarity(proto) > self.config.similarity_threshold_artistic:
                    return True
        return False

    def _is_valid_entity(self, token: Token) -> bool:
        # Ignore extremely common single-word tokens that spaCy sometimes mislabels.
        if self._get_lexeme_rank(token) < self.config.entity_generic_rank_threshold:
            return False
        return token.ent_type_ in self.config.valid_entity_types

    def _is_color(self, token_or_text: str | Token) -> bool:
        text = token_or_text if isinstance(token_or_text, str) else token_or_text.text
        text = text.strip().lower()

        # Direct CSS colour name lookup
        # Remove trailing punctuation (e.g., "red," -> "red")
        text = text.rstrip(".,;:!?\"')")

        # WebColors name lookup (CSS3 then CSS21)
        try:
            webcolors.name_to_hex(text, spec="css3")
            return True
        except ValueError:
            try:
                webcolors.name_to_hex(text, spec="css21")
                return True
            except ValueError:
                pass

        return False

    def _compute_centroid(self, words: Sequence[str]):
        """Compute mean vector for a list of seed words."""
        vectors = [
            self.nlp(word)[0].vector for word in words if self.nlp(word)[0].has_vector
        ]
        if not vectors:
            return np.zeros(self.nlp.meta.get("vectors", {}).get("width", 300))
        return np.mean(vectors, axis=0)

    def _is_concrete(self, token: Token) -> bool:
        if not token.has_vector:
            return False
        centroid_concrete_norm = np.linalg.norm(self._concrete_centroid)
        centroid_abstract_norm = np.linalg.norm(self._abstract_centroid)
        sim_concrete = token.vector.dot(self._concrete_centroid) / (
            token.vector_norm * centroid_concrete_norm
        )
        sim_abstract = token.vector.dot(self._abstract_centroid) / (
            token.vector_norm * centroid_abstract_norm
        )
        return sim_concrete - sim_abstract > 0.05  # margin

    # ------------------------------------------------------------------
    # Candidate phrases --------------------------------------------------
    # ------------------------------------------------------------------
    def _extract_candidates(self, doc: Doc) -> List[str]:
        chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
        tokens = [
            token.text.strip()
            for token in doc
            if token.pos_ in self.config.content_pos_tags and token.text.strip()
        ]
        return sorted(set(chunks + tokens), key=lambda x: -len(x))

    def _get_lexeme_rank(self, token: Token) -> int:
        """Return frequency rank of the token's lexeme or a large number if unknown."""
        lexeme = token.doc.vocab[token.text.lower()]
        return getattr(lexeme, "rank", 1_000_000)


# -------------------------------------------------------------------------
# Simple CLI for convenience (can be removed if undesired)
# -------------------------------------------------------------------------


def main():
    """Interactive CLI that prints JSON and a readable summary."""
    query = input("Enter your search query: ").strip()

    if not query:
        print("No input provided. Exiting.")
        return

    analyzer = NLPAnalyzer()
    result = analyzer.analyze_query(query)

    # Raw JSON dump
    print("\n=== Pure spaCy Linguistic Analysis ===")
    print(json.dumps(result, indent=2))

    # Pretty summary
    print("\n=== Structured Output ===")
    print(f"🎯 Main Objective: {result['main_objective']}")
    if result["sub_objectives"]:
        print(f"📋 Sub Objectives: {', '.join(result['sub_objectives'])}")

    for attr_type, values in result["attributes"].items():
        if not values:
            continue
        if attr_type == "entities":
            entity_strs = [f"{item['text']} ({item['label']})" for item in values]
            print(f"🏷️  {attr_type.title()}: {', '.join(entity_strs)}")
        elif attr_type == "format_types":
            print(f"📄 Format Types: {', '.join(values)}")
        elif attr_type == "colors":
            print(f"🎨 Colors: {', '.join(values)}")
        elif attr_type == "sizes":
            print(f"📏 Sizes: {', '.join(values)}")
        elif attr_type == "materials":
            print(f"🧾 Materials: {', '.join(values)}")
        elif attr_type == "shapes":
            print(f"🔷 Shapes: {', '.join(values)}")
        else:
            print(f"🏷️  {attr_type.title().replace('_', ' ')}: {', '.join(values)}")


if __name__ == "__main__":
    main()
