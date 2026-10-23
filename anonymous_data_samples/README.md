# Trait Evidence Graph data samples

This package contains two complete participant examples from RIIV (called IELTS in the original project files) and two from RecruitView. Each example includes face-redacted source videos, turn-aligned transcripts, reference evidence directions, reference reasoning, and an archived pipeline example. The examples illustrate the data format and evidence structure; they are not a random evaluation subset or a performance benchmark.

## Sample contents

| Sample | Topic examples | Turns | Videos |
|---|---|---:|---:|
| RIIV sample 01 | Cooking, leisure, and restaurants | 7 | 1 |
| RIIV sample 02 | Art, animals, bags, and birthdays | 12 | 1 |
| RecruitView sample 01 | Stress, performance, and competing priorities | 6 | 6 |
| RecruitView sample 02 | Risk, teamwork, mentoring, and interests | 6 | 6 |

Each sample directory contains:

- `interview.json`: questions, answers, relative video paths, time intervals in seconds, and media information.
- `media/`: videos with opaque head-region masks applied to every speaker, including interviewers. RIIV uses complete interview recordings; RecruitView uses per-turn recordings. Video resolution, frame counts, presentation times, and original audio payloads are preserved.
- `evidence_annotations.jsonl`: one reference annotation record per turn, covering all five traits, including `no_clear`.
- `reference_reasoning.json`: provided person–trait reference reasoning. These are reference texts used by the project, not measured personality scores or a guarantee of independent human validation.
- `pipeline_example.json`: archived Qwen2.5-Omni-7B Router decisions, Analyst directions, predicted graph nodes and edges, and final reasoning. Predictions are distinct from reference annotations. Graph aggregate statistics are not part of this structure-only snapshot.

The five trait codes are O (Openness), C (Conscientiousness), E (Extraversion), A (Agreeableness), and N (Neuroticism).

## Video and timestamp alignment

All paths are relative to their sample directory. In RIIV, question and answer intervals refer to the complete `media/interview.mp4` timeline. In RecruitView, intervals refer to the corresponding `media/turn_NNN.mp4`. A null question interval means the source has no aligned spoken-question interval; the textual question is still supplied. The full videos may contain material outside the annotated intervals.

## Reference annotations and predictions

Reference annotations contain `support_high`, `support_low`, `mixed`, or `no_clear`. The `is_evidence` flag is false exactly when the direction is `no_clear`. These are evidence directions rather than participant-level trait labels. Reference reasoning may use `Turn`, `turn`, or `qa_id` references, all of which refer to the retained turn numbering.

Graph nodes and edges in `pipeline_example.json` belong to model predictions, so they can disagree with the reference annotations. A `cross_context_support` edge links same-pole directional predictions; a `conflict` edge links opposing poles. Node identifiers resolve to the sample's raw interview turns. An empty predicted graph remains a valid archived prediction. Model reasoning is preserved from the archived run rather than regenerated for this release.

## Repository anonymity and media content

Sample identifiers and video filenames are pseudonyms. Exported text replaces directly stated participant names with `[NAME]`. Server paths, machine names, account information, access tokens, training states, and reviewer records are not included. Faces are obscured using opaque masks covering head regions and their movement within each shot. Voices, spoken names, burned-in subtitles, clothing, and background context remain; face masking does not make participants fully anonymous. Original dataset and media rights and access terms remain applicable.

## Verify the package

From this directory run:

```bash
python3 verify_samples.py
```

The verification uses the Python standard library. It checks every listed file's SHA-256, sample counts, video references, time bounds, annotations, graph references, and repository path hygiene. All redacted videos passed full audio/video decoding. Audio packet payload hashes and audio/video presentation timestamps were checked against the unredacted source copies. `face_redaction_summary.json` records the release transformation.

## Interpretation of the masked videos

Evidence annotations, reference reasoning, and archived pipeline predictions come from the original unredacted interviews. Facial-expression evidence described in those records is obscured in this release. The masked package supports inspection of data format, transcripts, audio, and graph structure; it does not reproduce all original visual evidence.

## Upload with Git LFS

The package includes `.gitattributes` for MP4 files. Install Git LFS, then run the following in the destination Git repository before adding these samples:

```bash
git lfs install
git lfs track "*.mp4"
git add .gitattributes README.md DATA_FORMAT.md manifest.json face_redaction_summary.json verify_samples.py RIIV RecruitView
git commit -m "Add dataset samples"
```

Use the anonymous repository's author identity for commits. The destination hosting service must support delivery of the LFS media objects for viewers to download the masked videos.
