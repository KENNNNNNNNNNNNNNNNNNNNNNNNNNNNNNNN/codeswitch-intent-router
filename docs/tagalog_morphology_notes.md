# Tagalog verb forms, Taglish verbs, slang, and raw/daw: notes for the tagger

Research notes written for tagger v2 (November 2026). Summaries are in our own
words; sources are listed at the end. Anything marked **(check)** comes from
general knowledge rather than a cited source and should be confirmed by a
native speaker.

## 1. Why this matters for the tagger

The tagger decides a word's language by its **root** (guideline Rule 1). To
find the root it has to undo Tagalog affixes and reduplication. If it can strip
`nagluluto` down to `luto` but `luto` is not in any word list, the word stays
UNK. So two things are needed: rules for the affixes, and enough root words.

## 2. Aspect: how Tagalog marks when an action happens

Tagalog verbs change for **aspect** (whether the action is done, ongoing, or
not yet started) rather than tense. Aspect is shown by a prefix or infix and/or
by repeating the first syllable of the root.

| Affix class | Completed (done) | Progressive (ongoing) | Contemplated (not yet) | Example root |
|---|---|---|---|---|
| mag- | nag + root | nag + CV + root | mag + CV + root | luto: nagluto, nagluluto, magluluto |
| -um- | -um- in root | -um- + CV + root | CV + root | kain: kumain, kumakain, kakain |
| ma- | na + root | na + CV + root | ma + CV + root | tulog: natulog, natutulog, matutulog |
| -in- (object focus) | -in- in root | -in- + CV + root | CV + root + -in | sabi: sinabi, sinasabi, sasabihin |

CV = the first consonant and vowel of the root, repeated (`lu` in `luto` gives
`luluto`). For roots starting with a vowel, only the vowel repeats and a hyphen
is written: `aral` gives `nag-aaral`, `mag-aaral`.

**na- vs ma-:** `ma-` verbs and adjectives switch to `na-` in the completed and
progressive aspects (`matulog` / `natulog`). As adjectives, `ma-` marks a
quality (`maganda`), and `na-` also marks something that happened to someone,
often by accident (`nahulog`, `na-late`, `na-shock`).

Other prefixes the tagger strips: `naka-/maka-` (able to, or managed to),
`nakaka-` (causing a feeling: `nakakapagod`), `magpa-/nagpa-` (have something
done), `pag-` (the act of), `ka-` (just finished, with repetition), `i-`
(object focus: `i-share`). Suffixes `-an/-han` and `-in/-hin` are also removed.

## 3. Taglish verbs: Tagalog affixes on English roots

English verbs and nouns are used as Tagalog verb roots, taking the same
affixes and the same syllable repetition. Common written forms **(check)**:

| Form | Pattern | Meaning |
|---|---|---|
| mag-review, nag-review | prefix + hyphen + English root | will / did review |
| nagre-review, magre-review | prefix + first syllable of root, repeated | is reviewing / will review |
| nagsho-shopping, nags-shopping, nagsha-shopping | the repeated part varies in spelling | is shopping |
| magpapa-check | magpa- (have it done) + repeated pa + root | will have it checked |
| nakaka-stress, na-shock, ma-late | adjective/feeling prefixes | stressful, got shocked, will be late |
| i-share, i-send, pina-check | object focus | share it, send it, had it checked |

How the tagger handles them: the part before the hyphen is read as a Tagalog
prefix, plus at most a short repeated piece that matches the start of the
root (`re` before `review`, `s`/`sho` before `shopping`) or repeats the prefix
(`pa` in `magpapa-`). The root after the hyphen decides the tag, so all of the
above are EN, while `nag-aaral` and `mag-usap` are TL.

## 4. Slang

Kept in `data/lexicon/slang.csv` with our own short meanings. Tagging follows
the guidelines:

- Slang built from Tagalog, or English respelled or reshaped the Filipino way,
  is **TL** (Rule 3): `lodi` (idol, read backwards), `werpa` (power),
  `petmalu` (malupet, backwards), `charot`, `jowa`, `marites`, `forda`.
- English slang used as is, is **EN**: `rizz`, `no cap`, `slay`, `ghosting`.
- Text-speak short forms take the language of the full word: `nmn` (naman),
  `kc` (kasi), `pde` (pwede) are TL.

Slang changes fast, and lists online are incomplete and sometimes dated. Every
row in the CSV must be reviewed by a native speaker before it is trusted.

## 5. raw/daw and rin/din

`daw/raw` mark reported speech ("they say", "reportedly"); `din/rin` mean "too,
also". The formal (KWF) rule picks the form from the **last sound of the word
before it**:

- after a **vowel** or the semivowels **w** and **y**: `raw`, `rin`
  (`maganda raw`, `ikaw rin`, `siya raw`)
- after any **other consonant**: `daw`, `din` (`bukas daw`, `ganoon din`)

In casual and informal writing the two forms are often swapped freely, so the
app shows this as a **suggestion for formal writing**, never as an error. One
source also mentions using the r-form after words ending in `d`; we leave that
case without a suggestion.

## Sources

- University of Hawaii Filipino program, grammar notes on aspect and on
  enclitics (daw/raw, din/rin): https://hawaii.edu/filipino/Grammar_Topics/Grammar_2-1.html,
  https://hawaii.edu/filipino/Grammar_Topics/Grammar_4-3.html
- Wikipedia, "Tagalog grammar" (affix chart, reduplication): https://en.wikipedia.org/wiki/Tagalog_grammar
- Wiktionary, "Appendix:Tagalog verbs": https://en.wiktionary.org/wiki/Appendix:Tagalog_verbs
- Wikipedia, "Taglish" (English words as Tagalog verb roots): https://en.wikipedia.org/wiki/Taglish
- Filipino Journal, "'Din' versus 'Rin'": https://filipinojournal.com/?p=4610
- Open Exam Prep, CSE spelling and mechanics (rin/raw after vowel or w/y): https://open-exam-prep.com/study-guides/ph-cse-professional/sentence-paragraph/spelling-mechanics
- Kaiwa, "Filipino Gen Z slang phrases 2026": https://trykaiwa.com/blog/filipino-gen-z-slang-phrases-2026
- Daily Tribune, "From '67' to 'clock it': internet slang Filipinos can't stop searching" (Sept 2026): https://tribune.net.ph/2026/09/25/from-67-to-clock-it-internet-slang-filipinos-cant-stop-searching
- Ling, "Gen Z slang in Tagalog": https://ling-app.com/blog/gen-z-slang-in-tagalog/
