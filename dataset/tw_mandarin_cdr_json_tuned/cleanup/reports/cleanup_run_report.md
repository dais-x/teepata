# Cleanup Run Report

- Source sample files: 1000
- Cleaned sample files: 1000
- Before blocker counts: {'contains_patient_label': 403, 'duplicated_pause_event_count': 1000, 'event_stats_mismatch': 482, 'marker_event_sequence_mismatch': 704, 'missing_top_level_speech_rate_target': 1000, 'simplified_chinese_detected': 40, 'spoken_transcript_contains_markers': 51}
- After blocker counts: {}
- Manual clinical review items after cleanup: 242

## Strict Result

PASS: strict mechanical cleanup checks passed.

## Outputs

- `cleanup/cleaned_dataset/`
- `cleanup/cleaned_dataset/manifest.json`
- `cleanup/reports/before_audit_summary.json`
- `cleanup/reports/after_audit_summary.json`
- `cleanup/reports/changes.csv`
- `cleanup/reports/after_manual_review.csv`
