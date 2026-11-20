# Word-frequency lists

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
