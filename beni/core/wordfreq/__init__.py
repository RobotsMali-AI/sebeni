"""Wordfreq: DabaX + Distiller-checkpoint counts (bamfreq-style, not FastText)."""

from __future__ import annotations

import json
import glob
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Sequence

from beni.core.language import Language
from beni.core.morphotactic.dabax import get_dabax
from beni.core.morphotactic.dabax import resolve_baseline_files
from beni.utils import config as cfg


@dataclass
class WordfreqReport:
    """Surface / lemma / morpheme / stage histograms."""

    language: str
    checkpoint_id: Optional[str] = None
    surfaces: Dict[str, int] = field(default_factory=dict)
    lemmas: Dict[str, int] = field(default_factory=dict)
    morphemes: Dict[str, int] = field(default_factory=dict)
    stages: Dict[str, int] = field(default_factory=dict)
    misses: Dict[str, int] = field(default_factory=dict)
    n_tokens: int = 0
    n_sentences: int = 0

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "checkpoint_id": self.checkpoint_id,
            "n_tokens": self.n_tokens,
            "n_sentences": self.n_sentences,
            "surfaces": self.surfaces,
            "lemmas": self.lemmas,
            "morphemes": self.morphemes,
            "stages": self.stages,
            "misses": self.misses,
        }

    def write(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "wordfreq.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        for name, table in (
            ("surfaces.tsv", self.surfaces),
            ("lemmas.tsv", self.lemmas),
            ("morphemes.tsv", self.morphemes),
            ("stages.tsv", self.stages),
            ("misses.tsv", self.misses),
        ):
            lines = ["form\tcount"]
            for form, count in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"{form}\t{count}")
            (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _walk_morphemes(morpheme) -> List[str]:
    forms = [morpheme.form] if getattr(morpheme, "form", None) else []
    nested = getattr(morpheme, "morphemes", None) or []
    for child in nested:
        forms.extend(_walk_morphemes(child))
    return forms


def count_texts(
    texts: Iterable[str],
    lang: str,
    gram: Optional[Union[str, Path]] = None,
    ldict: Optional[Union[str, Path]] = None,
    checkpoint_id: Optional[str] = None,
) -> WordfreqReport:
    """Tokenize and analyze ``texts`` with DabaX against the current checkpoint.

    Parameters
    ----------
    texts : iterable of str
        Source sentences.
    lang : str
        ISO or Sebeni group code.
    gram, ldict : path, optional
        Distiller checkpoint files. Defaults to DabaX resolution.
    checkpoint_id : str, optional
        Recorded in the report (e.g. ``baseline_v3``).
    """
    group = Language.from_code(lang).group_code
    dabax = get_dabax(
        group,
        gram=gram,
        ldict=ldict,
        process=True,
        runtime_dir=cfg.get_workdir().runtime,
    )
    surfaces: Counter = Counter()
    lemmas: Counter = Counter()
    morphemes: Counter = Counter()
    stages: Counter = Counter()
    misses: Counter = Counter()
    n_sent = 0
    n_tok = 0

    for text in texts:
        if not text:
            continue
        try:
            sentences = dabax.loader(text)
        except Exception:
            continue
        for sent in sentences:
            n_sent += 1
            for tok in sent.tokens:
                n_tok += 1
                surfaces[tok.surface] += 1
                stages[str(tok.stage)] += 1
                try:
                    if int(tok.stage) == -1:
                        misses[tok.surface] += 1
                except (TypeError, ValueError):
                    pass
                if tok.analyses:
                    lemmas[tok.analyses[0].form] += 1
                    for morph in tok.analyses[0].morphemes or []:
                        for form in _walk_morphemes(morph):
                            morphemes[form] += 1
                else:
                    for form in tok.best_morphemes():
                        morphemes[form] += 1

    return WordfreqReport(
        language=group,
        checkpoint_id=checkpoint_id,
        surfaces=dict(surfaces),
        lemmas=dict(lemmas),
        morphemes=dict(morphemes),
        stages=dict(stages),
        misses=dict(misses),
        n_tokens=n_tok,
        n_sentences=n_sent,
    )


def _raw_paths(raw_inputs: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    inputs = [raw_inputs] if isinstance(raw_inputs, (str, Path)) else list(raw_inputs)
    paths: List[Path] = []
    for value in inputs:
        text = str(value)
        if any(char in text for char in "*?["):
            paths.extend(Path(path) for path in sorted(glob.glob(text, recursive=True)))
            continue
        path = Path(value)
        if path.is_dir():
            paths.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in {".txt", ".text"}
                )
            )
        elif path.is_file():
            paths.append(path)
    return paths


def _raw_texts(blob: str) -> List[str]:
    normalized = blob.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks = normalized.split("\n\n") if "\n\n" in normalized else normalized.splitlines()
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def count_raw_inputs(
    raw_inputs: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    languages: Optional[Sequence[str]] = None,
    default_lang: str = "bam",
    encoding: str = "utf-8-sig",
) -> Dict[str, WordfreqReport]:
    """Build per-language DabaX frequency maps from raw text files."""
    allowed = set(Language.group_codes(languages)) if languages else None
    grouped: Dict[str, List[str]] = {}
    paths = _raw_paths(raw_inputs)
    for path in paths:
        stem_known = bool(
            cfg.get_language_metadata(path.stem) or cfg.get_language_iso(path.stem)
        )
        stem_group = (
            Language.from_code(path.stem).group_code
            if stem_known
            else Language.from_code(default_lang).group_code
        )
        if allowed and len(allowed) == 1:
            group = next(iter(allowed))
        elif allowed and stem_group not in allowed:
            continue
        else:
            group = stem_group or Language.from_code(default_lang).group_code
        grouped.setdefault(group, []).extend(
            _raw_texts(path.read_text(encoding=encoding))
        )

    reports = {}
    for group, texts in grouped.items():
        gram, ldict, checkpoint_id = resolve_baseline_files(group)
        reports[group] = count_texts(
            texts,
            lang=group,
            gram=gram,
            ldict=ldict,
            checkpoint_id=checkpoint_id,
        )
    return reports
