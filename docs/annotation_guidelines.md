# Token-Level Annotation Guidelines (v0.2)

v0.2 adds Rules 7-9 for everyday text (tagger v2, `src/tagger_v2.py`).
Rules 1-6 are unchanged, so the Day 1-7 gold tags are still valid.

## Tags
- EN: English token
- TL: Tagalog token
- NE: language-independent token (excluded from density)
- UNK: the tagger could not decide (excluded from density). Only the tagger
  outputs UNK; human annotators always choose EN, TL or NE.

## Tokenization
- Split on whitespace. Strip punctuation (. , ? !).
- Curly apostrophes (’) count as straight ones (').
- Hyphenated tokens such as "i-request" stay as one token.
- Lowercase before comparing, but keep the original text.

## Rules
1. Affixed words: the ROOT decides the tag.
   "i-request", "Na-sign", "mag-enroll", "nagre-reflect" -> EN.
   "magbabayad" -> TL (Tagalog root).
2. Borrowed English words written in English spelling -> EN.
   "clearance", "tuition", "schedule", "enrollment" -> EN.
   This matches the sentence-level Rule A.
3. Filipinized or Spanish-origin words in Filipino spelling -> TL.
   "klase", "iskedyul", "semestre", "estudyante", "bangko" -> TL.
4. Names, brands, acronyms, numbers -> NE.
   "GCash", "TOR", "Math", "137" -> NE.
5. Function words follow their language:
   "ko", "na", "ba", "yung", "po", "ang", "ng" -> TL;
   "my", "I", "the", "to" -> EN.
6. Ambiguous token: tag by the more likely language and record it in the notes.
7. Contractions are split into their parts, and each part is tagged:
   "Ako'y" -> "Ako" (TL) + "'y" (= ay, TL); "Ako't" -> "Ako" + "'t" (= at);
   "Ano'ng" -> "Ano" + "'ng" (= ang); "sa'yo" -> "sa" + "'yo" (= iyo).
   Shortened words stay one token in their own language:
   "'to" (ito), "'di" (hindi), "'wag" (huwag) -> TL; "'cause" (because) -> EN.
   English apostrophe forms stay one EN token: "don't", "I'm", "registrar's".
8. A Tagalog prefix written apart from its root is TL: "makapag wait" ->
   "makapag" TL, "wait" EN.
9. "may": TL when it means "there is / has" ("may unpaid balance"),
   EN when it is the English modal ("May I ask").

## Density (defined in the next step)
Computed only over EN and TL tokens. NE tokens are ignored.