# Maintenance review — 2026-09-27

## Purpose and baseline

The chooser is a static browser app for comparing vLLM models against GPU memory and quantization compatibility. The checkout had one pre-existing edit in generated `data.js`; it was preserved. Shareable filter URLs serialize the GPU-count setting, including the `Any` value (`0`), but URL restoration used a truthiness fallback that converted `0` to the default one-GPU setting.

## Changes

`app.js` now restores only GPU counts present in the page's filter controls, preserving `0` and falling back to one GPU for malformed or unsupported values. The test harness accepts an initial search string and models the GPU-count buttons; `tests/logic.test.mjs` covers `0`, the maximum supported value, and invalid counts.

## Verification

Ran `npm test` offline: 44 passed, 0 failed, 0 skipped. No upstream sync or factcheck was run because those workflows fetch external catalog data.

## Remaining gaps

The generated model data was not refreshed or checked against live upstream sources in this offline review. The pre-existing generated-data edit remains intact.
