# Retained evaluation evidence

## Holdout8 (current)

[holdout8/](holdout8/) is the final evaluation of the integrated research changes on 2026-10-30: 760 files (440 images, 160 videos, 160 texts) from twenty originals per kind, source-disjoint from holdout7 and every earlier manifest (seed `2026103008`, same generator and encoder defaults). Media and sources are excluded, as for holdout7.

Scored **once** with the binary in `binary-sha.txt`: one cold local scan and one cold mixed-URL scan, `--thumbnails on`, URL run `keyframes`, all other settings default (`--remote-overlay on`, `--image-source-bank on`). Precision 1.000 in every category, zero hard-negative errors.

| Run | Duplicate direct / group recall | Variant recall | FrameOf direct / group | Time |
| --- | ---: | ---: | ---: | ---: |
| Local | 0.945 / 0.994 | 0.885 | 288/318 / 315/318 | 541 s |
| Mixed URL | 0.901 / 0.955 | 0.885 | remote 17/42 / 21/42 | 505 s |

The URL run fetched 26.8% of remote body bytes. Holdout7 and holdout8 use different sources, so differences between them are indicative, not a controlled comparison. No thresholds were tuned from these results. Three later fixes (remote continuation after a refused window, a stricter text template guard, documentation) are not reflected in these reports; they were verified on tuning corpora only.

## Holdout7

[holdout7/](holdout7/) records the designated final evaluation on 2026-10-29: 760 source-disjoint natural-source files, built from twenty licensed Wikimedia Commons images, twenty native Commons videos and twenty public-domain Wikisource works. It contains 440 images, 160 texts and 160 videos; truth covers 870 duplicate pairs (330 `frame_of`), 877 Variant pairs and thirty hard negatives.

The frozen corpus was **scored once**, with one cold local scan and one cold mixed-URL scan using default matching and `--thumbnails on`; the URL run used `keyframes`. Local duplicate direct/group recall was 0.907/0.974; mixed-URL recall was 0.841/0.929, both with duplicate precision 1.000. See [local-score.txt](holdout7/local-score.txt), [url-score.txt](holdout7/url-score.txt) and the [known limits](../README.md#known-limits). No thresholds were tuned from these results. These reports capture the frozen binary identified in `binary-sha.txt`, not subsequent fixes.

## Construction and regeneration

The construction command, with the recorded seed and archived source manifest, is:

```sh
eidolon-eval generate-sources ./source-manifest.json ./holdout7-regenerated --seed 2026102907
```

1. Restore source representations and original response captures using the manifest's `source_url`, `work_url` and recorded revision/provenance. Verify each source's `sha256` and any `raw_sha256`; hashes identify the required bytes, not a promise that an upstream URL stays unchanged.
2. For texts, reproduce the recorded UTF-8 work-body paragraph extraction, structurally excluding licence banners and navigation. The extracted text must match its recorded source hash; raw API JSON is not interchangeable with it.
3. In a working copy of the manifest, rebase `path` and `raw_capture` to the restored files. Run the command above with an empty output directory. The generator is offline and performs no acquisition.

Media can be regenerated from the hash-verified source representations using the same transformation matrix and seed. Exact encoded derivative hashes also depend on codec/tool versions. The original acquisition/extraction scripts and raw captures are not bundled here; the manifest retains their original paths and lineage hashes. This is an evidence archive, not a self-contained source download bundle.

## Retained files and size

Construction-lineage hashes identify the original construction artifacts; source and payload hashes identify media bytes. Scores and reports describe the recorded evaluation runs. `holdout-role.json` and `freeze-receipt.json` describe the **pre-evaluation** freeze; their `never_detector_scored`/`no_detector_evaluation` fields are historical, not the current score status. `eidolon-score.json` is the URL run's evaluator-owned score; both named score text files are retained to avoid confusing the two runs.

| Evidence | Size (bytes) |
| --- | ---: |
| `local-output.json` | 3,399,253 |
| `url-output.json` | 3,296,144 |
| `truth.json` | 680,111 |
| `source-manifest.json` | 70,143 |
| `eidolon-score.json` | 67,330 |
| `url-traffic.json` | 13,643 |
| `url-stderr.txt` | 9,695 |
| `url-score.txt` | 8,503 |
| `local-score.txt` | 8,103 |
| `freeze-receipt.json` | 3,578 |
| `counts.json` | 1,434 |
| `holdout-role.json` | 779 |
| `distinctness.json` | 571 |
| `local-stderr.txt` | 281 |
| `binary-sha.txt` | 183 |
| `frame-windows.json` | 129 |
| `registration.json` | 82 |
| `local-time.txt`, `url-time.txt` | 36 each |
| **19 evidence files total** | **7,560,034 (7.21 MiB)** |

The table lists the retained holdout7 file sizes; media and raw source captures are not bundled. The roughly 2.7 GB `media/` and `sources/` directories are deliberately excluded; no executable or model is included.
