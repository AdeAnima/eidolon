# eidolon-text

Local-only Unicode text duplicate, excerpt and template-variant detection. The public entry point is `find(files, ctx)`; inputs are read, never rewritten.

## Relations

- Byte-identical files remain exact duplicates in the CLI. NFKC, full case folding and whitespace-normalized equality produce `SameContent`, including CRLF and rewrapped saves.
- Complete texts that align after masking typed data fields produce `Variant` only with field/template evidence: a changed explicitly labelled identifier or amount, or a changed ID-like value in bounded fields of a shared HTML form with at least twenty unchanged visible words. Plain numbers, dates, footer years and visitor counts do not establish template evidence. Variants carry the evidence note **same template, different data**. They are related documents, not duplicates: the CLI excludes these edges from duplicate groups and `keep_hint` selection.
- Ordinary prose replacements remain `Edited` or `Similar` according to their measured five-word shingle Jaccard overlap. Directional excerpts retain their existing containment and length requirements.

Exact typed alignment masks numeric fields (including whole decimal amounts with dot/comma separators, numeric dates/times and mixed alphanumeric IDs), long hexadecimal codes, email addresses and HTTP(S) URL query values. Masking alone does not establish a template: ordinary prose with one or two changed numbers remains `Edited` or `Similar`, including prose wrapped in HTML. Alphabetic IDs count as data after an explicit code/reference/order/booking-number label, including labels in adjacent HTML cells. English and German month names and AM/PM are data only beside a numeric date/time component. Other text and punctuation must align across the complete normalized documents.

Identifier context includes `Receipt #`, `Invoice #`, `Order #`, `Rechnung #`, `Bestellung #`, `Receipt number`, `Invoice No.`, `Rechnung Nr.` and `Belegnummer`. A period marks an identifier-label abbreviation only after `No` or `Nr`; an ordinary sentence ending in “total.” is not an amount label. Amount context includes Amount/Total/Gesamt/Price/Preis and accepts currency symbols (`€`, `$`, `£`, `¥`) or codes (EUR/USD/GBP/JPY) between the label and number. A cents-only change retains the label on the whole amount, including dot/comma decimals.

HTML forms additionally admit short field replacements such as merchant/property names, streets, postcodes and weekdays. This requires a changed explicitly labelled identifier or numeric amount, at least twenty unchanged visible words, and alignment of every remaining field. Exact typed alignment can supply the anchor only with a changed ID-like value: an email address, long hexadecimal code containing a letter, URL query value, mixed letters/digits, or explicitly labelled identifier. Plain numeric/date-only changes cannot supply this anchor. Changed fields are bounded to eight words and 120 characters; at most four consecutive short fields on each side may split or merge before alignment resumes. Changed fields of at least three words ending in sentence punctuation, and longer prose blocks, are not field replacements. Without a changed ID/amount anchor or exact typed ID-like alignment, short word edits remain duplicates. Inline markup is ignored within these fields, and head/style/script content cannot supply the fixed-word proof.

An exact typed-data alignment with field context, or a guarded field-template alignment, has confidence 1; the evidence retains the unmasked Jaccard and containment measurements. Normalized equality is checked first, so an unchanged receipt never becomes a Variant. Typed alignment streams borrowed tokens and compares original data values, so unchanged labelled fields cannot supply the changed-field proof. HTML field alignment uses ranges into the normalized text rather than copied field strings. The `text.shingles/v4` feature key includes these per-file field ranges and proof metadata; older payloads are not reused because identifier/amount label recognition changed cached field anchors. Both documents must meet the existing minimum substantial-text requirement, and existing LSH/rare-anchor candidate generation still applies.

## HTML and boilerplate

HTML is processed as source text for candidate generation and shingle similarity, not as rendered prose. Markup, inline styles and repeated links therefore contribute alongside visible text. A shared receipt template can have high shingle overlap even when amounts, dates, IDs, merchant names and addresses differ. Shingle overlap alone is not sufficient duplicate evidence: typed-data alignment or the guarded visible-field comparison routes these pairs to the non-duplicate Variant channel.

## Checks

```sh
cargo fmt -p eidolon-text -- --check
cargo clippy --offline -p eidolon-text --all-targets -- -D warnings
cargo nextest run --offline -p eidolon-text
```

Synthetic public-boundary tests cover ordinary prose with a changed year, one or two changed figures, decimal figures and HTML wrapping; multi-block HTML footer year/date/visitor updates; explicitly labelled plain-text receipt amounts and identifiers; cents-only dot/comma and currency-prefix changes; English/German labels; changed HTML receipt data; different merchant names/addresses/booking data; bounded field-layout changes; amount-only anchors; missing-anchor rejection; email/hex/code/query fields; genuine prose edits even beside a changed booking ID; CRLF duplicates; and cached results. The CLI integration regression additionally checks that template variants create no duplicate groups or keep hints, while a CRLF re-save groups only with its own receipt. No private receipts are stored in this crate.

The evaluator's file/pair `variant` property names transformations such as `light_edits`, `excerpt_10`, `rewrapped` and `crlf`; it is not the `Relation::Variant` channel. Existing small4 and corrected5 text truth contains no non-duplicate template-variant positives. Those corpora protect prose/excerpt duplicate recall; synthetic receipts protect the template distinction.
