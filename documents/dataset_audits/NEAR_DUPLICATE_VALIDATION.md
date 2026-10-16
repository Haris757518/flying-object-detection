# Near-Duplicate Validation Summary

## Experiment Setup
- **Methodology**: Structural Similarity Index Measure (SSIM) combined with dimensional heuristics.
- **Dataset**: 100 randomly sampled pairs with pHash distance <= 5.

## Category Breakdown
- **A (Exact Duplicate)**: 35
- **B (Resized/Compressed)**: 35
- **C (Minor Augmentation/Weather)**: 16
- **D (Different image, same object)**: 12
- **E (Clearly different image)**: 2

## Overall Metrics
- **Percentage A+B (True Duplicates)**: 70.00%
- **Percentage C (Augmented Overlap)**: 16.00%
- **Percentage D+E (False Positives)**: 14.00%
- **Estimated Precision of pHash <= 5**: 86.00%

## Recommendation
A significant fraction of the sampled pairs are false positives (distinct images mapped to low pHash distances). A threshold of `pHash <= 5` is **TOO AGGRESSIVE** and destroys legitimate training variance.

**Proposed Two-Stage Verification Rule:**
1. **Stage 1 (Candidate Generation):** `pHash_distance <= 5`
2. **Stage 2 (Verification):** Validate candidates using SSIM > 0.70 or ORB feature matching.
