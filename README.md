# wolof-normalizer

Normalise Wolof text written in French-style / SMS spelling into the **official orthography** (CLAD alphabet).

```
$ wolof-normalize "Nanga def ? Mangui fi rek, dieuredieuf"
Na nga def ? Maa ngi fi rekk, jërëjëf
```

Most Wolof online (social media, messaging) is written with French spelling habits: `kh` for `x`, `ou` for `u`, `gn` for `ñ`, `dieu` for `jë`… This makes search, NLP tools and corpus building much harder. This small library maps these spellings back to the standard form.

## How it works

For each word, from safest to least safe:

1. **Lexicon** of known variants, including multi-word expressions (`nanga def` → `na nga def`), in [`data/lexicon.tsv`](src/wolof_normalizer/data/lexicon.tsv).
2. Words already in the **official vocabulary** ([`data/vocab.txt`](src/wolof_normalizer/data/vocab.txt)) are kept.
3. **Grapheme rules** (`tch`→`c`, `kh`→`x`, `ch`→`c`, `dj`→`j`, `gn`/`ny`→`ñ`, `ou`→`u`, `eu`→`ë`, `di`+vowel→`j`, final consonant gemination). A rule-based correction is applied **only if the result exists in the vocabulary**, to avoid breaking names and French words.
4. Otherwise the word is left unchanged.

Case and punctuation are preserved.

## Install & use

```bash
pip install -e ".[dev]"
```

```python
from wolof_normalizer import normalize, normalize_with_changes

normalize("dafa beug wakh")          # 'dafa bëgg wax'
normalize_with_changes("kham nga ko ?")
# ('xam nga ko ?', [Change(original='kham', normalized='xam', reason='rules')])
```

Command line:

```bash
wolof-normalize "khale yi"            # xale yi
cat messages.txt | wolof-normalize -v # -v prints each change on stderr
```

## Tests

```bash
pytest
```

## Contributing

The lexicon and vocabulary are small for now and are the easiest place to help: add one `variant<TAB>official` line to `lexicon.tsv`, or one official word per line to `vocab.txt`.

## Roadmap

- Grow the lexicon and vocabulary from public Wolof resources
- Evaluation set (SMS sentences ↔ official spelling) and accuracy metrics
- Handle vowel length and accents (`a`/`à`/`aa`, `e`/`é`/`ë`)
- Learned model (character-level) for words outside the vocabulary
- Publish on PyPI and as a Hugging Face Space demo

## License

MIT, Baye Lahad Mbacke
