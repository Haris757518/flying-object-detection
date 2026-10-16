# Unified Airborne Object Detection Project

One project root with separated code, data, documentation, research, and results.

| Folder | Contents |
| --- | --- |
| code/hackathon_app | Previous four-class dashboard, modules, checkpoints and required demo assets |
| code/research_prototype | New desktop/web prototype and hybrid model definitions |
| code/dataset_tools | Dataset repair utility; use explicit paths described below |
| datasets/combined_dataset | Active corrected six-class dataset and metadata |
| datasets/source_material | Original AOD/MBAOD files, exports, notebooks and source archives |
| datasets/legacy_merged | Previous merged dataset, preserved for reference |
| documents/reports | Hackathon and research Word reports |
| documents/dataset_audits | Earlier dataset reports and duplicate review samples |
| documents/research_results | Earlier generated research-results narrative |
| documents/organization | Relocation plan, completed-move journal and verification |
| research/papers | Five reference PDFs |
| results/research_prototype | Simulated/hard-coded research tables and figures; not measured evidence |
| results/historical_hackathon | Historical benchmark JSON and generated test media |
| archives | Existing ZIPs, repair cache and original configuration/code snapshots |

## Current state

Dataset: 55,409 images (38,786 train / 8,311 validation / 8,312 test).
Class IDs: 0 airplane, 1 bird, 2 helicopter, 3 kite, 4 parachute, 5 UAV.
Training configuration: datasets/combined_dataset/data.yaml.

The research results supplied with the prototype are generated from constants.
Do not present them as measured training/evaluation results. New six-class
training, actual evaluation, and integration of trained hybrid components remain
future work. Organizing folders does not complete those stages.

The original applications remain separate so that their behavior can be reviewed
before integration. Checkpoints and required input videos stay with their apps
to preserve existing relative lookups. No datasets or model weights were removed.

## Launch existing applications

Use a Python environment containing the appropriate application dependencies.
From PowerShell in this project root:

    .\launch_hackathon.ps1
    .\launch_research.ps1

These launchers select each app's working directory. They do not install packages
or certify model correctness. The research launcher defaults to the web UI;
use -Module 8 for its desktop UI. Its old training utilities still reference
DS/comb DSF and must be reviewed and configured before training. Do not run them
against the corrected dataset until the known leakage issues are fixed.

To run only the existing dataset verifier after relocation:

    python code/dataset_tools/repair_merged_dataset.py verify --root "D:\ai\datasets\source_material" --output "D:\ai\datasets\combined_dataset"

## Historical paths and existing ZIPs

Manifest source paths, historical reports, archived repair caches and existing
ZIP contents retain their original D:\ai paths. They are provenance records,
not current training configuration. Use project.json and the relocation plan
to resolve their new locations. Original ZIPs were not rebuilt. Update the path
in a runtime copy of data.yaml when transferring the dataset to another machine.

Dataset group checks reduce leakage but cannot prove full recording independence
where MBAOD source-video identifiers are unavailable. Preserve that limitation.

## Next work

1. Select and train a real six-class baseline on the frozen dataset splits.
2. Replace simulated result generation with actual prediction-based evaluation.
3. Train and integrate external classifiers and the selected GAN approach.
4. Correct tracking, class mapping and validation leakage issues.
5. Consolidate the selected implementation and update reports using measured data.
