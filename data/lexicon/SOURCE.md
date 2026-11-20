# Word lists used by the v2 tagger

`tl_freq.txt` (Tagalog) and `en_freq.txt` (English) are used by the v2 token
tagger (`src/tagger_v2.py`). Each line is `word count`.

- Source: FrequencyWords by Hermit Dave, https://github.com/hermitdave/FrequencyWords
  (2018 lists, built from OpenSubtitles 2018, http://opus.nlpl.eu/OpenSubtitles2018.php)
- Licence of the word lists: **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/).
  These two files are a derived work and stay under CC BY-SA 4.0. This licence
  covers only the files in this folder, not the rest of the repository.
- Changes: lowercased; kept alphabetic words (apostrophes and hyphens allowed);
  Tagalog list keeps words seen at least twice. Rebuild with
  `python experiments/build_lexicon.py`.

Known limitation: subtitles contain names and some English words inside the
Tagalog list (and the reverse). The tagger compares relative frequencies in
both lists, so a word like "the" that appears in Tagalog subtitles is still
tagged English.

## slang.csv

Filipino and English slang and text-speak, compiled in November 2026 from the
sources listed in `docs/tagalog_morphology_notes.md` and general knowledge.
Meanings are written in our own words. Rows marked `source = general (check)`
and every row with `status = draft` still need review by a native speaker.
Set `status` to `ok` after checking a row, or to `remove` to make the tagger
ignore it. Multi-word phrases are kept for reference; the tagger tags their
words one by one.

## tl_dict.txt (optional, built by you)

Built by `python experiments/build_tl_dictionary.py <file>` from the Tagalog
data at https://kaikki.org/dictionary/Tagalog/ (machine-readable English
Wiktionary, extracted with wiktextract). Wiktionary text is licensed under
**CC BY-SA**; this word list is a derived work under the same licence.
Proper names and words whose etymology says "from English" are left out. The
tagger only consults it for words that are not in the frequency lists.
