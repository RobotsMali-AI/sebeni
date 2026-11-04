# `beni.core.language`

```python
from beni.core.language import Language, parse_lang_codes

Language.from_code("mlq").group_code   # 'mku'
Language.group_codes("bam,mku,dtm")    # ['bam', 'mku', 'dtm']
parse_lang_codes(["bam", "mku,dtm"])    # ['bam', 'mku', 'dtm']
Language.from_code("dyu")              # Language(iso='dyu', group_code='bam', ...)
```

`from_code` looks the code up as a group code or variant in
`beni/utils/res/language_metadata.json`; a code that is not listed there keeps
itself as its group (name from the ISO 639-3 table). An empty code means `bam`.
`parse_lang_codes` lower-cases and splits on `,` or `;`.

::: beni.core.language.Language
    options:
      members:
        - from_code
        - group_codes
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: true
      show_if_no_docstring: false

::: beni.core.language.parse_lang_codes
    options:
      show_root_heading: true
      heading_level: 2
