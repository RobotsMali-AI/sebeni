# Languages

Mali's 2023 constitution lists **13 national languages**. Sebeni scores Φ and
Distiller checkpoints **per group code**. Completions must declare that row's
group in JSON `lang`.

| Language | ISO you may pass | Sebeni group | Notes |
| --- | --- | --- | --- |
| Bambara (Bamanankan) | bam, bm, dyu | **bam** | Packaged baseline + largest raw split |
| Bomu | bmq | **bmq** | Packaged baseline |
| Bozo | boz, bze, bzx, boo | **boz** | Family grouped; packaged baseline |
| Dogon | dtm, dts, djm, dti, … (19 variants) | **dtm** | Family grouped; packaged baseline |
| Fula / Fulani | ful, ff, fuv, fuh, ffm, fue, fub, fuq, fui, fuf, fuc | **ful** | Packaged baseline |
| Hassaniya | mey | **mey** | Packaged baseline |
| Kassonke (Xaasongaxango) | kao | **kao** | Packaged baseline |
| Mamara | myk | **myk** | Packaged baseline |
| Maninka | mku, mlq, emk, msc, mwk | **mku** | Packaged baseline under `baselines/mku/` (`mlq/` is still accepted as a fallback directory) |
| Seynara / Senoufo | spp, seq, sep, sev, … | **spp** | Senufo group; packaged baseline |
| Songhay | ses, khq, hmb | **ses** | Packaged baseline |
| Soninke | snk | **snk** | Packaged baseline |
| Tamasheq | taq, tmh | **taq** | Packaged baseline |

The full variant lists live in `beni/utils/res/language_metadata.json`.

Aliases such as `bm` → `bam` and `mlq` → `mku` are resolved by
`Language.from_code`. Pass ISO or group to `--lang`.

```bash
sebeni init --lang bam --lang mku --lang dtm --lang mey -w ./runs/mali-001
```

```yaml
data:
  default_lang: bam          # unlabeled rows only
  languages: [bam, mku, dtm] # filter + prompt scope
```

```python
from beni.core.language import Language, parse_lang_codes

Language.from_code("mlq").group_code   # 'mku'
Language.group_codes("bam,mku,dtm")    # ['bam', 'mku', 'dtm']
parse_lang_codes(["bam", "mku,dtm"])
```

Packaged train texts live under `beni/data/raw/{group}.txt`. A code with no
baseline (for example `bbo`, Konabéré, in the packaged raw split) uses
Distiller **scratch bootstrap**. A code outside the metadata file must be a
valid ISO 639-3 code: Distiller warns, uses it as its own group, and fetches
its metadata from Wikidata on first use (`beni.utils.hf_lang`, network
required; cached in `{working_dir}/data/language_metadata.json`). Any other
code raises. Adding a new language: [use case 4](use-cases.md#4-language-with-no-packaged-grammar-scratch-bootstrap).

Thirteen-language GRPO is [use case 9](use-cases.md#9-all-13-national-languages).
