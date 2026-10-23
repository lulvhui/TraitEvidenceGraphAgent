# Sample data format

Each `sample_id` is local to this package. Turn IDs are strings preserving the original within-interview numbering. Sample files do not expose the original participant identifiers.

## Interview

`interview.json` contains `sample_id`, `dataset`, `timestamps_relative_to`, `turns`, and `media`. Each turn provides `question`, `answer_text`, `video_path`, and nullable question/answer intervals `[start, end]` in seconds. `media` records contain relative paths, SHA-256 values, file sizes, durations, audio/video presence, resolution, frame count, frame rate, and face-redaction metadata. All released videos are transcoded face-masked derivatives; audio payloads and presentation times match the unredacted source copies.

## Evidence annotations

Each line of `evidence_annotations.jsonl` is one turn with exactly O, C, E, A, and N entries. Every trait has an `is_evidence` boolean and a `direction` enum: `support_high`, `support_low`, `mixed`, or `no_clear`. All annotated turn–trait pairs are retained, including non-evidence pairs.

## Reference reasoning

`reference_reasoning.json` provides one reasoning text per trait. These supplied references are separate from model predictions. No continuous or categorical participant personality score is exported.

## Pipeline example

`router_predictions` contains turn-wise five-trait `yes`, `no_clear`, or `no` relevance decisions. `analyst_predictions` contains directions for inspected turn–trait pairs. `graphs` contains predicted evidence nodes and unordered edge endpoints identified by turn IDs; the two edge relations are `cross_context_support` and `conflict`, with weights in [0,1]. Each trait's `traits` entry contains the archived final reasoning text. This is a qualitative illustration rather than an executable training record or a refreshed evaluation run.

## Integrity manifest

`manifest.json` lists the four samples, media totals, and SHA-256/size values for every other package file, including the verifier and hidden Git configuration. The manifest itself is excluded from its own file index.
