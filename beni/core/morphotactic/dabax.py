import re
import warnings
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, Any, List, Tuple
from xml.dom import minidom
from beni.utils import config as cfg
from beni.core.language import Language

warnings.filterwarnings("ignore", category=FutureWarning)

BASELINES = cfg.DATA_DIR / "baselines"


def _daba_modules():
    """Import CLI mparser/ntgloss without pulling Daba GUI (wxPython)."""
    try:
        from daba import mparser, ntgloss
    except ImportError as exc:
        raise ImportError(
            "daba is required for DabaX. Install the CLI parser (no wxPython on "
            "the default extra): pip install 'daba>=0.9.5'. Upstream: "
            "https://github.com/maslinych/daba (GPLv2+). The mlsftwrs CLI-only "
            "fork is the intended sebeni dependency."
        ) from exc
    return mparser, ntgloss


class DabaX(object):
    """Daba processor wrapping CLI ``mparser`` (no Wx)."""

    __GRAM_BASELINE = "baseline.gram"
    __DICT_BASELINE = "baseline.dict"

    def __init__(
        self,
        lang,
        text=None,
        gram=None,
        ldict=None,
        tokenizer="default",
        runtime_dir: Optional[Union[str, Path]] = None,
        process: bool = False,
    ):
        """Set up a Daba processor.

        If ``process`` is True, load baselines from the working dir (Distiller
        checkpoints) rather than packaged ``beni/data/baselines``.
        """
        identity = Language.from_code(str(lang)) if lang else Language.from_code("bam")
        self.lang = identity.group_code
        self.text = text
        self.tokenize_type = tokenizer
        self.runtime_dir = Path(runtime_dir) if runtime_dir else cfg.get_workdir().runtime
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        packaged = cfg.resolve_packaged_baseline_dir(self.lang)
        wd_base = cfg.get_workdir().baselines / self.lang
        if process:
            baseline_dir = wd_base if wd_base.exists() else (packaged or wd_base)
        else:
            baseline_dir = packaged or (BASELINES / self.lang)

        self.gramf = gram if gram else Path(baseline_dir) / self.__GRAM_BASELINE
        self.dictf = ldict if ldict else Path(baseline_dir) / self.__DICT_BASELINE
        self.dl, self.gr, self.tokenizer = self.setup_parser()

    def setup_parser(self, runtime_dir: Optional[Union[str, Path]] = None):
        """Load Dict and Gram Loader from CLI ``mparser`` (headless)."""
        mparser, _ntgloss = _daba_modules()
        runtime = Path(runtime_dir) if runtime_dir else self.runtime_dir
        runtime.mkdir(parents=True, exist_ok=True)
        dl = mparser.DictLoader(runtimedir=str(runtime), verbose=False)
        dl.addfile(str(self.dictf))

        gr = mparser.GrammarLoader(runtimedir=str(runtime))
        gr.load(str(self.gramf))

        self.set_tokenizer(self.tokenize_type)
        return (dl, gr, self.tokenizer)

    def set_tokenizer(self, tokenizer: str):
        """Set Tokenizer for parsing."""
        mparser, _ntgloss = _daba_modules()
        self.tokenizer = mparser.Tokenizer()
        self.tokenizer.use_method(tokenizer)

    def mparser(self):
        """Daba Mparser Object."""
        mparser, _ntgloss = _daba_modules()
        return mparser.Processor(
            dictloader=self.dl, grammarloader=self.gr, tokenizer=self.tokenizer, detone=True
        )

    def __ensure_eos(self, text: str, eos: str='</s>', newline: str='\n') -> str:
        """
        Ensure EOS symbol in text, else replace new lines with EOS.
        """
        text = text.replace(newline, eos)
        if not text.endswith(eos):
            text += eos
        return text

    def text_processor(self, text: str, eos: str='</s>', newline: str='\n'):
        """ Process Text """
        text = self.__ensure_eos(text, eos=eos, newline=newline)
        texts = [t.strip() for t in text.split(eos) if t.strip()]
        mp = self.mparser()

        return mp.parse(texts)

    def unwrap_gloss(self, gloss) -> dict:
        """
        Recursively Unwrap Gloss Object to Dictionary
        """
        result = {
            "form": getattr(gloss, 'form', ''),
            "ps": getattr(gloss, 'ps', ()),
            "gloss": getattr(gloss, 'gloss', ''),
        }

        morphemes = getattr(gloss, 'morphemes', ())

        if morphemes:
            result["morphemes"] = [self.unwrap_gloss(m) for m in morphemes]

        return result

    def extract_morphology(self, text, **kw):
        """Retrieve Text Morphological Information"""

        if kw:
            parsed = self.text_processor(text, **kw)
        else:
            parsed = self.text_processor(text)

        sentences = {}

        for sent in parsed:
            
            for st in sent:
                boundary, tokens = st
                sentence = boundary.value.strip()
                sentences[sentence] = []

                for tok in tokens:
                    tok_tup = tok.as_tuple()
                    if(tok_tup[0] == "w"): # Token]
                        sentences[sentence].append(self.analyze_token(tok_tup[1]))
                    else:
                        ## Assuming Punctuation:
                        pass
        return sentences

    def analyze_token(self, token_data: Tuple[str, str, list]) -> dict:
        """
        Process a single token entry into a structured tree representation.
        """

        token, stage, glosses = token_data

        return {"token": token, "stage": stage, "analyses": [self.unwrap_gloss(g) for g in glosses]}

    def add_lexical_info(self, parent_elem, data):
        """
        Recursively adds form, ps, gloss, and morphemes to an XML element.
        """
        # Add <form>
        form_elem = ET.SubElement(parent_elem, "form")
        form_elem.text = data.get("form", "")
        
        # Add <ps> (Can be a list of strings)
        for p in data.get("ps", []):
            ps_elem = ET.SubElement(parent_elem, "ps")
            ps_elem.text = p
            
        # Add <gloss>
        gloss_elem = ET.SubElement(parent_elem, "gloss")
        gloss_elem.text = data.get("gloss", "")
        
        # Recursively handle nested <morphemes>
        morphemes = data.get("morphemes")
        if morphemes:
            morphemes_elem = ET.SubElement(parent_elem, "morphemes")
            for morph_data in morphemes:
                morpheme_elem = ET.SubElement(morphemes_elem, "morpheme")
                self.add_lexical_info(morpheme_elem, morph_data)

    def json_to_xml(self, json_data: dict) -> List:
        """
        Converts the linguistic JSON data to an XML ElementTree structure.
        """
        # Root element
        # corpus = ET.Element("corpus")
        corpus = []

        # Iterate through sentences
        for sentence_text, tokens in json_data.items():
            sentence_elem = ET.Element("sentence")
            sentence_elem.set("text", sentence_text)
            
            # Iterate through tokens (annotations)
            for token_obj in tokens:
                annotation_elem = ET.SubElement(sentence_elem, "annotation")
                annotation_elem.set("token", token_obj.get("token", ""))
                annotation_elem.set("stage", str(token_obj.get("stage", "")))
                
                # Iterate through analyses
                for analysis_data in token_obj.get("analyses", []):
                    analysis_elem = ET.SubElement(annotation_elem, "analysis")
                    self.add_lexical_info(analysis_elem, analysis_data)

            xml_str = ET.tostring(sentence_elem, encoding="utf-8")
            parsed_str = minidom.parseString(xml_str)
            corpus.append(parsed_str.documentElement.toprettyxml(indent="\t"))

        return corpus
    
    def loader(self, text: str):
        """Convert text: raw string into structured Sentence objects."""
        from beni.core.compute import Sentence, Token, Analysis, Morpheme
        
        raw = self.extract_morphology(text)

        def _parse_ps(ps):
            if ps is None: return []
            if isinstance(ps, str): return [ps]
            return list(ps)[0] if list(ps) else ""  # FIXME: yield single PS element.

        def _build_morpheme(md):
            nested = None
            if "morphemes" in md and md["morphemes"]:
                nested = [_build_morpheme(m) for m in md["morphemes"]]
            return Morpheme(
                form=md.get("form", ""),
                ps=_parse_ps(md.get("ps")),
                gloss=md.get("gloss", ""),
                morphemes=nested
        )

        def _build_analysis(ad):
            morphemes = [_build_morpheme(m) for m in ad.get("morphemes", [])]
            return Analysis(
                form=ad.get("form", ""),
                ps=_parse_ps(ad.get("ps", None)),
                gloss=ad.get("gloss", ""),
                morphemes=morphemes
            )

        def _build_token(td):
            analyses = [_build_analysis(a) for a in td.get("analyses", [])]
            return Token(
                surface=td.get("token", td.get("surface", "")),
                stage=td.get("stage", -1),
                analyses=analyses
            )

        sentences = []
        
        for text, tokens_data in raw.items():
            # unwrap accidental double-list wrapping            
            while isinstance(tokens_data, list) and len(tokens_data) == 1 and isinstance(tokens_data[0], list):
                tokens_data = tokens_data[0]

            tokens = [_build_token(td) for td in tokens_data]
            sentences.append(Sentence(text=text, lang=self.lang, tokens=tokens))

        return sentences

    def parse(self, text: str= None):
        """
        Extracts morphological information from text and returns it in (str) XML format.
        """
        text = text if text else self.text

        if(text is None):
            raise ValueError("No text provided")
        
        parsed = self.extract_morphology(text)
        return self.json_to_xml(parsed)

    def __str__(self):
        return f"{__name__} - {self.lang} \n - {self.tokenizer} \n - {self.dictf} \n - {self.gramf}"
