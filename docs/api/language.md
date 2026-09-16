# `beni.core.language`

```python
from beni.core.language import Language, parse_lang_codes

Language.from_code("mlq").group_code   # 'mku'
Language.group_codes("bam,mku,dtm")    # ['bam', 'mku', 'dtm']
parse_lang_codes(["bam", "mku,dtm"])
```

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
