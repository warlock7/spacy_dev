from __future__ import annotations

from typing import Any, Dict, List, Sequence

from spacy.tokens import Doc, Token


class SemanticAttributeDetector:
    """Generic detector that flags tokens similar to a small prototype set."""

    def __init__(
        self,
        nlp_pipeline,
        prototypes: Sequence[str],
        similarity_threshold: float,
        category_name: str,
    ) -> None:
        self._nlp = nlp_pipeline
        self.category_name = category_name
        self._threshold = similarity_threshold
        self._prototype_tokens = [self._nlp(term)[0] for term in prototypes]

    def matches(self, token: Token) -> bool:
        if not token.has_vector:
            return False
        return any(
            token.similarity(prototype) >= self._threshold
            for prototype in self._prototype_tokens
        )


class AttributeManager:
    """Owns attribute detectors and high-level categorisation logic."""

    def __init__(self, analyzer: "NLPAnalyzer") -> None:  # type: ignore "NLPAnalyzer" is defined later
        self._analyzer = analyzer
        cfg = analyzer.config
        nlp = analyzer.nlp
        self._semantic_detectors: List[SemanticAttributeDetector] = [
            SemanticAttributeDetector(
                nlp,
                cfg.size_prototypes,
                cfg.similarity_threshold_size,
                "sizes",
            ),
            SemanticAttributeDetector(
                nlp,
                cfg.material_prototypes,
                cfg.similarity_threshold_material,
                "materials",
            ),
            SemanticAttributeDetector(
                nlp,
                cfg.shape_prototypes,
                cfg.similarity_threshold_shape,
                "shapes",
            ),
        ]

        # POS-aware classifier for grammatical refinements
        self._pos_classifier = POSAwareClassifier(analyzer)
        self._context_classifier = ContextFallbackClassifier(analyzer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def categorize_attributes(
        self,
        doc: Doc,
        main_obj: str | None,
        sub_objs: List[str],
    ) -> Dict[str, Any]:
        analyzer = self._analyzer
        cfg = analyzer.config
        attributes: Dict[str, List] = {
            "adjectives": [],
            "descriptors": [],
            "modifiers": [],
            "actions": [],
            "format_types": [],
            "entities": [],
            "colors": [],
            "sizes": [],
            "materials": [],
            "shapes": [],
        }
        content_words = {w.lower() for w in ([main_obj] if main_obj else [])} | {
            s.lower() for s in sub_objs
        }
        compound_mods = analyzer._extract_compound_modifiers(doc)
        tokens_in_comp = analyzer._get_tokens_in_compounds(compound_mods)

        for token in doc:
            token_text = token.text
            token_lower = token_text.lower()
            if token_lower in tokens_in_comp:
                continue

            # Colours
            if analyzer._is_color(token):
                attributes["colors"].append(token_text)
                continue

            best_category: str | None = None
            best_confidence: float = 0.0

            # Semantic detectors
            for detector in self._semantic_detectors:
                if detector.matches(token):
                    best_category, best_confidence = detector.category_name, 0.9
                    break

            # POS-aware layer
            if best_confidence < 0.75:
                pos_category, pos_conf = self._pos_classifier.classify(token)
                if pos_conf > best_confidence:
                    best_category, best_confidence = pos_category, pos_conf

            # Format type check
            if (
                best_confidence < 0.75
                and token.pos_ in cfg.content_pos_tags
                and analyzer._is_format_descriptor(token)
            ):
                best_category, best_confidence = "format_types", 0.8

            # Context fallback
            if best_confidence < 0.5:
                ctx_category, ctx_conf = self._context_classifier.classify(token, doc)
                if ctx_conf > best_confidence:
                    best_category, best_confidence = ctx_category, ctx_conf

            if best_category:
                attributes[best_category].append(token_text)
                continue

            # Format types
            if token.pos_ in cfg.content_pos_tags and analyzer._is_format_descriptor(
                token
            ):
                attributes["format_types"].append(token_text)
                continue

            # POS categories
            if token.pos_ == "ADJ":
                attributes["adjectives"].append(token_text)
            elif token.pos_ == "ADV":
                attributes["descriptors"].append(token_text)
            elif token.pos_ == "VERB" and token.tag_ == "VBG":
                attributes["actions"].append(token_text)
            elif token.pos_ == "VERB" and token.tag_ in cfg.past_participle_tags:
                if analyzer._is_artistic_descriptor(token):
                    attributes["descriptors"].append(token_text)
                else:
                    attributes["actions"].append(token_text)
            elif (
                token.dep_ in cfg.modifier_deps
                and token_lower not in content_words
                and not analyzer._is_format_descriptor(token)
                and not analyzer._is_color(token)
            ):
                attributes["modifiers"].append(token_text)

            # Entities
            if token.ent_type_ and analyzer._is_valid_entity(token):
                attributes["entities"].append(
                    {
                        "text": token_text,
                        "type": token.ent_type_,
                        "label": analyzer.nlp.vocab.strings[token.ent_type_],
                    }
                )

        if compound_mods:
            attributes["modifiers"].extend(compound_mods)

        # Remove main and sub objective words from attributes
        def _filter(tokens: List[str]) -> List[str]:
            return [t for t in tokens if t.lower() not in content_words]

        filtered_attributes: Dict[str, Any] = {}
        for name, vals in attributes.items():
            if not vals:
                continue
            if name == "entities":
                # For entities keep those whose text isn't main/sub objective
                filtered = [e for e in vals if e["text"].lower() not in content_words]
                if filtered:
                    filtered_attributes[name] = filtered
            else:
                uniq = list(set(_filter(vals)))
                if uniq:
                    filtered_attributes[name] = uniq

        finalized = filtered_attributes
        return finalized


# -----------------------------------------------------------------------------
# Objective (Main/Sub) Classifier
# -----------------------------------------------------------------------------


class ObjectiveClassifier:
    """Selects main and sub objectives from content candidates using layered scoring."""

    def __init__(self, analyzer: "NLPAnalyzer") -> None:  # type: ignore
        self._analyzer = analyzer
        cfg = analyzer.config
        self._subject_object_deps = set(cfg.subject_object_deps)

        # Container-category centroid built from a few seed words
        container_seeds = cfg.container_prototypes
        import numpy as np

        self._container_centroid = np.mean(
            [
                analyzer.nlp(w)[0].vector
                for w in container_seeds
                if analyzer.nlp(w)[0].has_vector
            ],
            axis=0,
        )
        self._container_centroid_norm = (
            float(np.linalg.norm(self._container_centroid)) or 1.0
        )

        # Theme prototypes for vector centroid (birthday, christmas, halloween, wedding)
        self._theme_proto_tokens = [
            analyzer.nlp(word)[0]
            for word in ["birthday", "christmas", "halloween", "wedding"]
        ]
        # Pre-compute centroid vector
        if self._theme_proto_tokens and self._theme_proto_tokens[0].has_vector:
            import numpy as np

            self._theme_centroid = np.mean(
                [tok.vector for tok in self._theme_proto_tokens if tok.has_vector],
                axis=0,
            )
            self._theme_centroid_norm = np.linalg.norm(self._theme_centroid)
        else:
            self._theme_centroid = None
            self._theme_centroid_norm = 1.0

    # Public API
    def choose(self, candidates: List[Dict[str, Any]]) -> tuple[str | None, List[str]]:
        if not candidates:
            return None, []

        # Compute max importance for normalization
        max_importance = max(c["importance"] for c in candidates) or 1

        def confidence(cand: Dict[str, Any]) -> float:
            """Combine several signals to produce 0-1 confidence."""
            conf = cand["importance"] / max_importance  # 0-1 baseline

            # Penalise format words heavily for objective selection
            if cand["is_format"]:
                conf *= 0.3

            # If this is a theme word acting as compound, force high confidence
            if cand["dep"] == "compound" and self._is_theme(cand["token"]):
                return 1.2  # guaranteed top score

            # Boost grammatical subjects / objects
            if cand["dep"] in self._subject_object_deps:
                conf += 0.2

            # Boost compound attached to a conjunct head (e.g., 'panda' in 'tiger and panda icon')
            if cand["dep"] == "compound" and cand["token"].head.dep_ == "conj":
                conf += 0.15

            # Penalise ROOT tokens that are semantically generic containers
            if cand["dep"] == "ROOT" and self._is_container(cand["token"]):
                conf = min(conf, 0.4)

            return min(conf, 1.0)

        # Assign confidence to each candidate
        for cand in candidates:
            cand["confidence"] = confidence(cand)

        # Main objective = highest confidence (ties broken by list order)
        main_candidate = max(candidates, key=lambda c: c["confidence"])
        main_text = main_candidate["text"]
        main_token = main_candidate["token"]
        main_is_theme = self._is_theme(main_token)

        # Sub-objectives: next best non-format words with confidence ≥ 0.8
        sub_objectives: List[str] = []
        for cand in sorted(candidates, key=lambda c: c["confidence"], reverse=True):
            if (
                cand is not main_candidate
                and not cand["is_format"]
                and cand["confidence"] >= 0.8  # stricter threshold
                and not (main_is_theme and cand["token"].i == main_token.head.i)
            ):
                sub_objectives.append(cand["text"])
                if len(sub_objectives) >= self._analyzer.config.max_sub_objectives:
                    break

        return main_text, sub_objectives

    # ------------------ theme detection ------------------
    def _is_theme(self, token: Token) -> bool:
        """Return True if token looks like an event/occasion word."""
        if token.ent_type_ in {"EVENT", "DATE"}:
            return True
        if not token.has_vector or self._theme_centroid is None:
            return False
        sim = token.vector.dot(self._theme_centroid) / (
            token.vector_norm * self._theme_centroid_norm
        )
        return sim >= 0.55

    def _is_container(self, token: Token) -> bool:
        """Check if token resembles a generic container noun via vector similarity."""
        if not token.has_vector:
            return False
        sim = token.vector.dot(self._container_centroid) / (
            token.vector_norm * self._container_centroid_norm
        )
        return sim >= 0.5


# -----------------------------------------------------------------------------
# POS-Aware Classifier
# -----------------------------------------------------------------------------


class POSAwareClassifier:
    """Refines token categories based on Part-of-Speech and dependency context."""

    def __init__(self, analyzer: "NLPAnalyzer") -> None:  # type: ignore
        self._analyzer = analyzer
        cfg = analyzer.config
        self._past_participle_tags = set(cfg.past_participle_tags)

    def classify(self, token: Token) -> tuple[str | None, float]:
        """Return (category, confidence). Confidence range 0–1."""
        # Gerund (VBG) – ongoing action noun
        if token.pos_ == "VERB" and token.tag_ == "VBG":
            return "actions", 0.9

        # Past participle artistic verbs → descriptors
        if (
            token.pos_ == "VERB"
            and token.tag_ in self._past_participle_tags
            and self._analyzer._is_artistic_descriptor(token)
        ):
            return "descriptors", 0.85

        # Adjectives → adjectives
        if token.pos_ == "ADJ":
            return "adjectives", 0.8

        # Concrete nouns serving as subject/object → modifiers (or potential sub-objective)
        if (
            token.pos_ == "NOUN"
            and token.dep_ in {"nsubj", "nsubjpass", "dobj", "pobj"}
            and self._analyzer._is_concrete(token)
        ):
            return "modifiers", 0.75

        return None, 0.0


# -----------------------------------------------------------------------------
# Context Fallback Classifier
# -----------------------------------------------------------------------------


class ContextFallbackClassifier:
    """Assigns tokens based on similarity to the overall sentence context."""

    def __init__(self, analyzer: "NLPAnalyzer") -> None:  # type: ignore
        self._analyzer = analyzer

    def classify(self, token: Token, doc: Doc) -> tuple[str | None, float]:
        if not token.has_vector or not doc.has_vector or token.vector_norm == 0:
            return None, 0.0
        sim = token.similarity(doc)
        if sim >= 0.6:
            return "modifiers", sim
        return None, sim
