import json
import logging
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import requests
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
from dotenv import load_dotenv
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import Interaction, LearningThread, RequestJob

load_dotenv()
logger = logging.getLogger(__name__)


# -----------------------------
# Configuration
# -----------------------------

MAX_PAPER_CONCEPTS = 12
MIN_PAPER_CONCEPTS = 8
MAX_PAPER_RELATIONSHIPS = 10
MIN_PAPER_RELATIONSHIPS = 6

# Used only for merging concept labels, NOT for creating relations.
# Keep this high to avoid false shared nodes.
AUTO_MERGE_SIMILARITY_THRESHOLD = 0.88
WEAK_RELATED_SIMILARITY_THRESHOLD = 0.78

SYSTEM_PROMPT = """
You are an academic research assistant for graduate students.
You extract compact, paper-local knowledge graphs from academic paper text.
Return only valid JSON when requested.

Critical rules:
- Extract concepts and relationships from the CURRENT paper only.
- Do not decide whether concepts are shared across papers.
- Do not invent concepts or relationships.
- It is acceptable to return fewer relationships if the text does not support them.
""".strip()


ALLOWED_RELATIONS = {
    "supports",
    "requires",
    "uses",
    "improves",
    "evaluates",
    "compares with",
    "extends",
    "causes",
    "part of",
    "covered in",
    "missing from notes",
    "relates to",
}

STRONG_CROSS_PAPER_RELATIONS = {
    "supports",
    "requires",
    "uses",
    "improves",
    "evaluates",
    "extends",
    "causes",
    "part of",
}

BAD_CONCEPTS = {
    "i", "we", "you", "they", "it", "this", "that", "these", "those",
    "the", "a", "an", "with", "without", "additionally", "however", "while",
    "unlike", "across", "through", "by", "for", "in", "on", "of", "and", "or",
    "paper", "study", "approach", "method", "methods", "framework", "system",
}

STOPWORDS_FOR_SIMILARITY = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "from", "using", "based", "level", "system", "systems", "method", "methods",
    "framework", "frameworks", "approach", "approaches", "model", "models",
}


# -----------------------------
# LLM call and parsing
# -----------------------------

def call_genai(prompt):
    url = os.getenv("GENAI_API_URL")
    api_key = os.getenv("GENAI_API_KEY")
    model = os.getenv("GENAI_MODEL", "llama3.1:latest")

    if not url or not api_key:
        raise ValueError("Missing GENAI_API_URL or GENAI_API_KEY in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.1,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


def clean_json_text(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    return cleaned


def default_result(message="The input was too limited to generate a reliable analysis."):
    return {
        "big_picture_summary": message,
        "concepts": [],
        "paper_concepts": [],
        "relationships": [],
        "paper_relationships": [],
        "alignment": {
            "covered_concepts": [],
            "missing_concepts": [],
            "weakly_covered_concepts": [],
        },
        "blind_spots": [],
        "concept_sources": {},
        "note_concepts": [],
        "possible_related_concepts": [],
    }


def parse_llm_json_response(text):
    try:
        parsed = json.loads(clean_json_text(text))
    except Exception:
        logger.error("Could not parse LLM JSON response: %s", text)
        return default_result("The AI response could not be parsed as structured JSON.")

    parsed.setdefault("big_picture_summary", "")
    parsed.setdefault("paper_concepts", parsed.get("concepts", []))
    parsed.setdefault("paper_relationships", parsed.get("relationships", []))
    parsed.setdefault("concepts", parsed.get("paper_concepts", []))
    parsed.setdefault("relationships", parsed.get("paper_relationships", []))
    parsed.setdefault("alignment", {})
    parsed["alignment"].setdefault("covered_concepts", [])
    parsed["alignment"].setdefault("missing_concepts", [])
    parsed["alignment"].setdefault("weakly_covered_concepts", [])
    parsed.setdefault("blind_spots", [])
    parsed.setdefault("concept_sources", {})
    parsed.setdefault("note_concepts", [])
    parsed.setdefault("possible_related_concepts", [])
    return parsed


# -----------------------------
# Normalization and validation
# -----------------------------

def normalize_relation_label(label):
    label = str(label or "relates to").strip().lower()

    label_map = {
        "support": "supports",
        "supported by": "supports",
        "require": "requires",
        "requires for": "requires",
        "use": "uses",
        "used for": "uses",
        "used by": "uses",
        "improve": "improves",
        "improves on": "improves",
        "evaluate": "evaluates",
        "evaluated by": "evaluates",
        "compare with": "compares with",
        "compares to": "compares with",
        "comparison": "compares with",
        "extend": "extends",
        "extends to": "extends",
        "cause": "causes",
        "leads to": "causes",
        "component of": "part of",
        "is part of": "part of",
        "covers": "covered in",
        "covered by": "covered in",
        "missing": "missing from notes",
        "not covered": "missing from notes",
        "related to": "relates to",
        "thematically related to": "relates to",
        "weakly relates to": "relates to",
    }

    label = label_map.get(label, label)
    return label if label in ALLOWED_RELATIONS else "relates to"


def _normalize_for_match(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _concept_variants(concept):
    raw = str(concept).strip()
    variants = {raw}

    for inside in re.findall(r"\(([^)]+)\)", raw):
        variants.add(inside)
        if inside.lower().endswith("s"):
            variants.add(inside[:-1])

    no_paren = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    if no_paren:
        variants.add(no_paren)

    words = re.findall(r"[A-Za-z]+", no_paren or raw)
    if len(words) >= 2:
        acronym = "".join(w[0] for w in words if w).upper()
        if len(acronym) >= 2:
            variants.add(acronym)
            variants.add(acronym + "s")

    more = set()
    for item in variants:
        item = str(item).strip()
        if not item:
            continue
        if item.lower().endswith("s") and len(item) > 3:
            more.add(item[:-1])
        else:
            more.add(item + "s")
    variants |= more

    return [v for v in variants if str(v).strip()]


def concept_supported_by_text(concept, text):
    if not concept or not text:
        return False

    normalized_text = f" {_normalize_for_match(text)} "
    if not normalized_text.strip():
        return False

    for variant in _concept_variants(concept):
        normalized_variant = _normalize_for_match(variant)
        if normalized_variant and f" {normalized_variant} " in normalized_text:
            return True

    return False


def is_good_concept(concept):
    text = str(concept or "").strip()
    norm = _normalize_for_match(text)

    if not text or not norm:
        return False
    if norm in BAD_CONCEPTS:
        return False
    if len(norm) < 3:
        return False
    if norm.isdigit():
        return False

    tokens = norm.split()
    if len(tokens) == 1 and tokens[0] in BAD_CONCEPTS:
        return False

    return True


def clean_concept_label(concept):
    text = str(concept or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'()[]{}")
    return text


def validate_current_paper_result(result, paper_text):
    """
    LLM can propose candidates, but code decides what survives.
    - Concepts must be reasonable and appear in the current paper text.
    - Relationships must have both endpoints in the validated current-paper concept set.
    - Per-paper KG is compact: max 8-12 concepts, max 6-10 relationships.
    """
    raw_concepts = result.get("paper_concepts", result.get("concepts", [])) or []
    clean_concepts = []
    seen = set()

    for concept in raw_concepts:
        concept = clean_concept_label(concept)
        key = concept.lower()
        if not is_good_concept(concept):
            continue
        if key in seen:
            continue
        if not concept_supported_by_text(concept, paper_text):
            continue
        clean_concepts.append(concept)
        seen.add(key)
        if len(clean_concepts) >= MAX_PAPER_CONCEPTS:
            break

    clean_concept_keys = {c.lower(): c for c in clean_concepts}
    relationships = []
    rel_seen = set()

    for rel in result.get("paper_relationships", result.get("relationships", [])) or []:
        if not isinstance(rel, dict):
            continue

        source = clean_concept_label(rel.get("source", ""))
        target = clean_concept_label(rel.get("target", ""))
        relation = normalize_relation_label(rel.get("relation", "relates to"))
        evidence = str(rel.get("evidence", "")).strip()

        if not source or not target or source.lower() == target.lower():
            continue
        if not is_good_concept(source) or not is_good_concept(target):
            continue
        if not concept_supported_by_text(source, paper_text):
            continue
        if not concept_supported_by_text(target, paper_text):
            continue

        # Endpoints must be part of the compact per-paper concept list.
        if source.lower() not in clean_concept_keys or target.lower() not in clean_concept_keys:
            continue

        # Evidence is useful but may be empty. If given, it must be from the current paper.
        if evidence and not any(
            _normalize_for_match(v) in _normalize_for_match(evidence)
            for v in [source, target]
        ):
            evidence = ""

        key = (source.lower(), relation, target.lower())
        if key in rel_seen:
            continue

        relationships.append({
            "source": clean_concept_keys[source.lower()],
            "relation": relation,
            "target": clean_concept_keys[target.lower()],
            "evidence": evidence,
            "scope": "within_paper",
        })
        rel_seen.add(key)

        if len(relationships) >= MAX_PAPER_RELATIONSHIPS:
            break

    result["paper_concepts"] = clean_concepts
    result["paper_relationships"] = relationships
    result["concepts"] = clean_concepts[:]
    result["relationships"] = relationships[:]
    return result


# -----------------------------
# Concept merge / provenance
# -----------------------------

def concept_tokens(concept):
    norm = _normalize_for_match(re.sub(r"\s*\([^)]*\)", "", str(concept)))
    tokens = []
    for token in norm.split():
        if token in STOPWORDS_FOR_SIMILARITY:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def token_similarity(a, b):
    ta = concept_tokens(a)
    tb = concept_tokens(b)
    if not ta or not tb:
        return 0.0

    sa = set(ta)
    sb = set(tb)
    jaccard = len(sa & sb) / max(1, len(sa | sb))
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(jaccard, seq)


def choose_canonical(existing_canonical_labels, new_label):
    """
    Merge only highly similar labels. This prevents false shared nodes.
    Used for node canonicalization only, never for relation creation.
    """
    if not existing_canonical_labels:
        return new_label, None, 0.0

    best_label = None
    best_score = 0.0
    new_norm = _normalize_for_match(new_label)

    for label in existing_canonical_labels:
        label_norm = _normalize_for_match(label)

        if new_norm == label_norm:
            return label, label, 1.0

        # Acronym / parenthesis variants.
        variants = {_normalize_for_match(v) for v in _concept_variants(label)}
        if new_norm in variants:
            return label, label, 1.0

        score = token_similarity(label, new_label)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= AUTO_MERGE_SIMILARITY_THRESHOLD:
        return best_label, best_label, best_score

    return new_label, best_label, best_score


def get_existing_project_context(thread):
    interactions = Interaction.objects.filter(thread=thread).order_by("created_at")
    summaries = []

    for item in interactions:
        try:
            data = json.loads(item.response)
            summaries.append({
                "summary": data.get("big_picture_summary", ""),
                "concepts": data.get("concepts", []),
                "paper_concepts": data.get("paper_concepts", data.get("concepts", [])),
                "relationships": data.get("relationships", []),
                "paper_relationships": data.get("paper_relationships", []),
                "blind_spots": data.get("blind_spots", []),
                "possible_related_concepts": data.get("possible_related_concepts", []),
                "current_paper_title": data.get("current_paper_title", ""),
                "current_user_notes": data.get("current_user_notes", ""),
                "paper_items": data.get("paper_items", []),
            })
        except Exception:
            continue

    return summaries


def extract_paper_title_from_user_input(user_input):
    match = re.search(r"^Paper title:\s*(.+)$", str(user_input or ""), flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def build_paper_items(existing_context, current_result, job):
    """Build a compact one-row-per-paper list for the Listed Paper View."""
    items = []

    for index, old in enumerate(existing_context, start=1):
        title = old.get("current_paper_title") or f"Paper {index}"
        concepts = old.get("paper_concepts", old.get("concepts", [])) or []
        notes = old.get("current_user_notes", "") or ""

        items.append({
            "paper_number": index,
            "title": title,
            "concepts": concepts,
            "notes": notes,
        })

    current_number = len(existing_context) + 1
    title = extract_paper_title_from_user_input(job.user_input) or f"Paper {current_number}"

    items.append({
        "paper_number": current_number,
        "title": title,
        "concepts": current_result.get("paper_concepts", current_result.get("concepts", [])) or [],
        "notes": job.user_notes or "",
    })

    return items


def merge_project_result(existing_context, new_result):
    """
    Deterministic project-level merge.
    - Each paper keeps a compact paper_concepts list.
    - The full project KG can grow without a concept count limit.
    - Shared source labels are computed later from paper_concepts only.
    - Similar labels can be canonicalized only when similarity is high.
    """
    canonical_labels = []
    label_map = {}
    possible_related = []

    def add_label(label):
        label = clean_concept_label(label)
        if not is_good_concept(label):
            return None

        canonical, nearest, score = choose_canonical(canonical_labels, label)
        label_map[label.lower()] = canonical

        if canonical not in canonical_labels:
            canonical_labels.append(canonical)
        elif canonical != label:
            label_map[label.lower()] = canonical

        if nearest and canonical == label and WEAK_RELATED_SIMILARITY_THRESHOLD <= score < AUTO_MERGE_SIMILARITY_THRESHOLD:
            possible_related.append({
                "concept_a": nearest,
                "concept_b": label,
                "similarity": round(score, 3),
                "status": "possible_related_not_merged",
            })

        return canonical

    # Add old paper-local concepts first, then new paper-local concepts.
    for old in existing_context:
        for concept in old.get("paper_concepts", old.get("concepts", [])):
            add_label(concept)

    for concept in new_result.get("paper_concepts", new_result.get("concepts", [])):
        add_label(concept)

    def map_concept(label):
        label = clean_concept_label(label)
        if not label:
            return ""
        key = label.lower()
        if key in label_map:
            return label_map[key]
        return add_label(label) or label

    relationships = []
    rel_seen = set()

    def add_relationship(rel, default_scope="within_paper"):
        if not isinstance(rel, dict):
            return

        source = map_concept(rel.get("source", ""))
        target = map_concept(rel.get("target", ""))
        relation = normalize_relation_label(rel.get("relation", "relates to"))
        evidence = str(rel.get("evidence", "")).strip()
        scope = str(rel.get("scope", default_scope)).strip() or default_scope

        if not source or not target or source.lower() == target.lower():
            return

        key = (source.lower(), relation, target.lower())
        if key in rel_seen:
            return

        relationships.append({
            "source": source,
            "relation": relation,
            "target": target,
            "evidence": evidence,
            "scope": scope,
        })
        rel_seen.add(key)

    for old in existing_context:
        for rel in old.get("paper_relationships", old.get("relationships", [])):
            add_relationship(rel, default_scope="within_paper")

    for rel in new_result.get("paper_relationships", new_result.get("relationships", [])):
        add_relationship(rel, default_scope="within_paper")

    blind_spots = []
    blind_seen = set()

    for old in existing_context:
        for blind in old.get("blind_spots", []):
            if not isinstance(blind, dict):
                continue
            concept = clean_concept_label(blind.get("concept", ""))
            key = concept.lower()
            if concept and key not in blind_seen:
                blind_spots.append(blind)
                blind_seen.add(key)

    for blind in new_result.get("blind_spots", []):
        if not isinstance(blind, dict):
            continue
        concept = clean_concept_label(blind.get("concept", ""))
        key = concept.lower()
        if concept and key not in blind_seen:
            blind_spots.append(blind)
            blind_seen.add(key)

    new_result["concepts"] = canonical_labels
    new_result["relationships"] = relationships
    new_result["blind_spots"] = blind_spots
    new_result["possible_related_concepts"] = possible_related[:20]

    return new_result


def add_concept_sources(existing_context, new_result):
    """
    Provenance is based ONLY on each paper's paper_concepts.
    Never infer concept_sources from relationship endpoints.
    This prevents false P1/P2/P3 shared badges.
    """
    concept_sources = defaultdict(set)
    visible_concepts = [str(c).strip() for c in new_result.get("concepts", []) if str(c).strip()]

    def canonical_for_visible(concept):
        if not concept:
            return None
        concept = clean_concept_label(concept)

        best = None
        best_score = 0.0
        for visible in visible_concepts:
            if _normalize_for_match(concept) == _normalize_for_match(visible):
                return visible
            score = token_similarity(concept, visible)
            if score > best_score:
                best_score = score
                best = visible

        if best_score >= AUTO_MERGE_SIMILARITY_THRESHOLD:
            return best
        return None

    def add_source(concept, paper_number):
        canonical = canonical_for_visible(concept)
        if canonical:
            concept_sources[canonical].add(paper_number)

    for idx, old in enumerate(existing_context):
        paper_number = idx + 1
        for concept in old.get("paper_concepts", old.get("concepts", [])):
            add_source(concept, paper_number)

    current_paper_number = len(existing_context) + 1
    for concept in new_result.get("paper_concepts", []):
        add_source(concept, current_paper_number)

    new_result["concept_sources"] = {
        concept: sorted(list(papers))
        for concept, papers in concept_sources.items()
    }
    return new_result


# -----------------------------
# Alignment / stats
# -----------------------------

def concept_appears_in_notes(concept, notes_text):
    if not notes_text or not str(notes_text).strip():
        return False

    normalized_notes = f" {_normalize_for_match(notes_text)} "
    if not normalized_notes.strip():
        return False

    for variant in _concept_variants(concept):
        normalized_variant = _normalize_for_match(variant)
        if not normalized_variant:
            continue
        if f" {normalized_variant} " in normalized_notes:
            return True

    return False


def enforce_note_alignment_from_user_notes(result, user_notes):
    # Compare notes mainly against current paper concepts, not the entire project history.
    concepts = [str(c).strip() for c in result.get("paper_concepts", []) if str(c).strip()]

    if not str(user_notes or "").strip():
        result["alignment"] = {
            "covered_concepts": [],
            "missing_concepts": concepts,
            "weakly_covered_concepts": [],
        }
        result["note_concepts"] = []
        return result

    covered = []
    missing = []

    for concept in concepts:
        if concept_appears_in_notes(concept, user_notes):
            covered.append(concept)
        else:
            missing.append(concept)

    result["alignment"] = {
        "covered_concepts": covered,
        "missing_concepts": missing,
        "weakly_covered_concepts": [],
    }
    result["note_concepts"] = covered
    return result


def enrich_blind_spots_from_alignment(result, user_notes, max_items=6):
    """
    Ensure Blind Spot Detection is visible even when the LLM returns few or no blind_spots.
    Blind spots are derived from current-paper concepts that are missing from the user's notes.
    """
    existing = []
    seen = set()

    for item in result.get("blind_spots", []) or []:
        if not isinstance(item, dict):
            continue
        concept = clean_concept_label(item.get("concept", ""))
        if not concept:
            continue
        key = concept.lower()
        if key in seen:
            continue
        existing.append({
            "concept": concept,
            "why_it_matters": str(item.get("why_it_matters", "")).strip() or "This concept appears in the paper but is not clearly reflected in your notes.",
            "suggested_next_step": str(item.get("suggested_next_step", "")).strip() or "Revisit the paper section where this concept appears and add a short note explaining its role.",
        })
        seen.add(key)

    alignment = result.get("alignment", {}) or {}
    missing = alignment.get("missing_concepts", []) or []
    weak = alignment.get("weakly_covered_concepts", []) or []

    has_notes = bool(str(user_notes or "").strip())

    for concept in list(missing) + list(weak):
        concept = clean_concept_label(concept)
        if not concept:
            continue
        key = concept.lower()
        if key in seen:
            continue

        if has_notes:
            why = f"'{concept}' appears as an important paper concept, but it is not clearly covered in your notes."
            step = f"Add one or two sentences explaining what '{concept}' means in this paper and how it connects to your listed concepts."
        else:
            why = f"'{concept}' is a key concept extracted from the paper, but no notes were provided for comparison."
            step = f"Write a brief note for '{concept}' so the system can compare your understanding with the paper content."

        existing.append({
            "concept": concept,
            "why_it_matters": why,
            "suggested_next_step": step,
        })
        seen.add(key)

        if len(existing) >= max_items:
            break

    result["blind_spots"] = existing[:max_items]
    return result

def enrich_blind_spots_from_paper_items(result, max_items=8):
    """
    Build Blind Spot Detection from the Listed Paper View.
    A blind spot = a paper concept that is not clearly covered in that paper's notes.
    This works even after removing the separate Alignment View.
    """
    blind_spots = []
    seen = set()

    paper_items = result.get("paper_items", []) or []

    for item in paper_items:
        paper_number = item.get("paper_number", "")
        title = item.get("title", f"Paper {paper_number}")
        notes = item.get("notes", "") or ""
        concepts = item.get("concepts", []) or []

        for concept in concepts:
            concept = clean_concept_label(concept)
            if not concept:
                continue

            key = f"{paper_number}:{concept.lower()}"
            if key in seen:
                continue

            # If notes are empty, every concept is a blind spot.
            # If notes exist, only concepts not appearing in notes become blind spots.
            if notes and concept_appears_in_notes(concept, notes):
                continue

            if notes:
                why = (
                    f"'{concept}' appears as an important concept in {title}, "
                    f"but it is not clearly reflected in your notes."
                )
                step = (
                    f"Add a short note explaining what '{concept}' means in this paper "
                    f"and how it connects to your listed concepts."
                )
            else:
                why = (
                    f"'{concept}' is listed as a key concept in {title}, "
                    f"but no notes were provided for this paper."
                )
                step = (
                    f"Write one or two sentences about '{concept}' so the system can compare "
                    f"your understanding with the paper content."
                )

            blind_spots.append({
                "concept": concept,
                "paper": f"Paper {paper_number}",
                "paper_title": title,
                "why_it_matters": why,
                "suggested_next_step": step,
            })

            seen.add(key)

            if len(blind_spots) >= max_items:
                result["blind_spots"] = blind_spots
                return result

    result["blind_spots"] = blind_spots
    return result

def get_project_stats(thread):
    interactions = Interaction.objects.filter(thread=thread).order_by("created_at")

    concept_set = set()
    blind_spot_set = set()

    for item in interactions:
        try:
            data = json.loads(item.response)

            for concept in data.get("concepts", []):
                if concept:
                    concept_set.add(str(concept).strip())

            for blind in data.get("blind_spots", []):
                if isinstance(blind, dict):
                    concept = blind.get("concept")
                else:
                    concept = blind

                if concept:
                    blind_spot_set.add(str(concept).strip())

        except Exception:
            continue

    papers_count = interactions.count()

    return {
        "papers_count": papers_count,
        "core_concepts_count": len(concept_set),
        "blind_spots_count": len(blind_spot_set),
        "next_paper_number": papers_count + 1,
    }


# -----------------------------
# Prompts
# -----------------------------

def build_current_paper_kg_prompt(paper_text, user_notes, total_papers_after_update):
    return f"""
You are extracting a compact knowledge graph from ONE current paper only.

Project paper count after this update: {total_papers_after_update}

Current paper content:
{paper_text}

User notes for this paper:
{user_notes}

Return ONLY valid JSON. Do not include markdown fences.

Required JSON structure:

{{
  "big_picture_summary": "A 5-8 sentence summary of the current paper and how it may fit into a broader reading project. Do not claim cross-paper overlap unless the current paper text itself supports it.",
  "paper_concepts": ["concept1", "concept2"],
  "paper_relationships": [
    {{
      "source": "concept1",
      "relation": "supports",
      "target": "concept2",
      "evidence": "short phrase or sentence from the current paper",
      "scope": "within_paper"
    }}
  ],
  "alignment": {{
    "covered_concepts": [],
    "missing_concepts": [],
    "weakly_covered_concepts": []
  }},
  "blind_spots": [
    {{
      "concept": "concept name",
      "why_it_matters": "short explanation",
      "suggested_next_step": "what the user should review next"
    }}
  ]
}}

Rules:
- Extract only {MIN_PAPER_CONCEPTS}-{MAX_PAPER_CONCEPTS} core concepts from the current paper.
- Extract only {MIN_PAPER_RELATIONSHIPS}-{MAX_PAPER_RELATIONSHIPS} relationships, if supported.
- If fewer than {MIN_PAPER_RELATIONSHIPS} relationships are clearly supported, return fewer.
- Do not include generic words such as: paper, study, method, framework, system, model, approach unless they are part of a named technical concept.
- Do not include pronouns or discourse words as concepts: we, I, this, that, the, additionally, however.
- Do not create shared labels such as P1/P2/P3. The code will compute sources later.
- Do not include concepts that are not explicitly present in the current paper text.
- Every relationship source and target must be listed in paper_concepts.
- Every relationship must be based on the current paper text only.
- Use only these relation labels: supports, requires, uses, improves, evaluates, compares with, extends, causes, part of, covered in, missing from notes, relates to.
- Prefer specific relation labels only when directly supported by the text.
- Include an evidence field for every relationship.
- Use "relates to" only when no clearer label is supported.
- Do not invent highly specific claims unsupported by the input.
""".strip()




def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


# -----------------------------
# Views
# -----------------------------

def explain_view(request):
    if request.GET.get("new") == "1":
        request.session.pop("current_thread_id", None)

    thread = None
    analyses_count = 0
    latest_result = None
    project_stats = {
        "papers_count": 0,
        "core_concepts_count": 0,
        "blind_spots_count": 0,
        "next_paper_number": 1,
    }

    projects = LearningThread.objects.all().order_by("-created_at")

    thread_id = request.session.get("current_thread_id")
    if thread_id:
        try:
            thread = LearningThread.objects.get(id=thread_id)
            analyses = Interaction.objects.filter(thread=thread).order_by("-created_at")
            analyses_count = analyses.count()
            project_stats = get_project_stats(thread)

            if analyses.exists():
                latest_result = json.loads(analyses.first().response)

        except Exception:
            thread = None
            request.session.pop("current_thread_id", None)

    return render(request, "explainer/explain.html", {
        "thread": thread,
        "has_project": thread is not None,
        "analyses_count": analyses_count,
        "latest_result": latest_result,
        "project_stats": project_stats,
        "projects": projects,
    })


@require_POST
def start_job(request):
    action_type = request.POST.get("action_type", "").strip()

    if action_type == "open_project":
        project_id = request.POST.get("project_id")

        try:
            thread = LearningThread.objects.get(id=project_id)
        except LearningThread.DoesNotExist:
            return JsonResponse({"error": "Project not found."}, status=404)

        request.session["current_thread_id"] = thread.id

        return JsonResponse({
            "status": "project_opened",
            "redirect": "/explainer/"
        })

    if action_type == "delete_project":
        project_id = request.POST.get("project_id")

        try:
            thread = LearningThread.objects.get(id=project_id)
        except LearningThread.DoesNotExist:
            return JsonResponse({"error": "Project not found."}, status=404)

        current_thread_id = request.session.get("current_thread_id")
        if current_thread_id and str(current_thread_id) == str(project_id):
            request.session.pop("current_thread_id", None)

        thread.delete()

        return JsonResponse({
            "status": "deleted",
            "redirect": "/explainer/?new=1"
        })

    if action_type == "create_project":
        project_name = request.POST.get("project_name", "").strip()
        conference = request.POST.get("conference", "").strip()

        if not project_name:
            return JsonResponse({"error": "Please enter a project name."}, status=400)

        title = project_name
        if conference:
            title = f"{project_name} — {conference}"

        thread = LearningThread.objects.create(title=title)
        request.session["current_thread_id"] = thread.id

        return JsonResponse({
            "status": "project_created",
            "redirect": "/explainer/"
        })

    paper_title = request.POST.get("paper_title", "").strip()
    paper_text = request.POST.get("paper_text", "").strip()
    user_notes = request.POST.get("user_notes", "").strip()
    paper_file = request.FILES.get("paper_file")

    if paper_file:
        try:
            pdf_text = extract_pdf_text(paper_file)
            if pdf_text:
                paper_text = f"{paper_text}\n\n{pdf_text}".strip()
        except Exception as e:
            logger.exception("Could not read uploaded PDF: %s", e)
            return JsonResponse({"error": "Could not read the uploaded PDF. Please paste the paper text instead."}, status=400)

    if action_type == "extract_kg":
        if not paper_text:
            return JsonResponse({"error": "Please provide paper content or upload a PDF to extract the KG."}, status=400)

        mode = "extract_kg"
        user_input = f"Paper title: {paper_title or 'Untitled Paper'}\n\nPaper content:\n{paper_text}"

    else:
        if not paper_text and not user_notes:
            return JsonResponse({"error": "Please provide paper content, upload a PDF, or enter your notes."}, status=400)

        mode = "knowledge_synthesis"
        user_input = f"Paper title: {paper_title or 'Untitled Paper'}\n\nPaper content:\n{paper_text}\n\nUser notes:\n{user_notes}"

    thread_id = request.session.get("current_thread_id")
    if thread_id:
        thread = LearningThread.objects.get(id=thread_id)
    else:
        thread = LearningThread.objects.create(title="Conference Knowledge Synthesis")
        request.session["current_thread_id"] = thread.id

    job = RequestJob.objects.create(
        thread=thread,
        mode=mode,
        paper_text=paper_text,
        user_notes=user_notes,
        user_input=user_input,
        status="queued",
        status_message="Request received."
    )

    return JsonResponse({"job_id": job.id})


def process_job(job):
    try:
        job.status = "analyzing"
        job.status_message = "Extracting a compact paper KG and merging it into the project graph."
        job.save()

        existing_context = get_existing_project_context(job.thread)
        total_papers_after_update = Interaction.objects.filter(thread=job.thread).count() + 1

        prompt = build_current_paper_kg_prompt(
            job.paper_text,
            job.user_notes,
            total_papers_after_update
        )

        raw_response = call_genai(prompt)
        current_result = parse_llm_json_response(raw_response)

        # Validate LLM output against current paper text and enforce per-paper limits.
        current_result = validate_current_paper_result(current_result, job.paper_text)

        # Merge compact paper KG into unbounded project-level KG.
        result = merge_project_result(existing_context, current_result)

        # Keep current paper concepts/relationships separately for provenance and future merges.
        result["paper_concepts"] = current_result.get("paper_concepts", [])
        result["paper_relationships"] = current_result.get("paper_relationships", [])
        result["big_picture_summary"] = current_result.get("big_picture_summary", "")

        # Metadata for compact listed-paper view.
        result["current_paper_title"] = extract_paper_title_from_user_input(job.user_input) or f"Paper {total_papers_after_update}"
        result["current_user_notes"] = job.user_notes or ""
        result["paper_items"] = build_paper_items(existing_context, current_result, job)

        # Notes alignment is deterministic and based on current paper concepts.
        if job.mode == "knowledge_synthesis":
            result = enforce_note_alignment_from_user_notes(result, job.user_notes)
        else:
            result["alignment"] = {
                "covered_concepts": [],
                "missing_concepts": [],
                "weakly_covered_concepts": [],
            }
            result["note_concepts"] = []

        # Make Blind Spot Detection robust by adding deterministic gaps from missing note coverage.
        result = enrich_blind_spots_from_alignment(result, job.user_notes)
        result = enrich_blind_spots_from_paper_items(result)
        
        # Compute P1/P2/shared labels from paper_concepts only.
        result = add_concept_sources(existing_context, result)

        job.response = json.dumps(result, ensure_ascii=False)
        job.status = "completed"
        job.status_message = "Analysis complete."
        job.save()

        Interaction.objects.create(
            thread=job.thread,
            user_input=job.user_input,
            mode=job.mode,
            response=job.response,
        )

    except Exception as e:
        logger.exception("Unexpected error while processing job: %s", e)
        job.status = "failed"
        job.error_message = str(e)
        job.status_message = "Analysis failed."
        job.save()


def job_status(request, job_id):
    try:
        job = RequestJob.objects.get(id=job_id)
    except RequestJob.DoesNotExist:
        return JsonResponse({"error": "Job not found."}, status=404)

    if job.status == "queued":
        process_job(job)
        job.refresh_from_db()

    result = None
    if job.response:
        try:
            result = json.loads(job.response)
        except Exception:
            result = default_result("Stored response could not be parsed.")

    analyses_count = Interaction.objects.filter(thread=job.thread).count()
    project_stats = get_project_stats(job.thread)

    return JsonResponse({
        "job_id": job.id,
        "status": job.status,
        "status_message": job.status_message,
        "result": result,
        "error_message": job.error_message,
        "analyses_count": analyses_count,
        "project_stats": project_stats,
    })


@require_POST
def confirm_job(request, job_id):
    return JsonResponse({
        "message": "Confirmation is not used in this workflow."
    })
