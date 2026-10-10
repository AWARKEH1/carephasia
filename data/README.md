# Data

## What is in this folder

| File | Content |
|---|---|
| `features_per_file.csv` | One row per transcript (463 rows): file id, diagnosis (`patient` / `control`) and the four computed features. No speech text. |
| `group_averages.csv` | Mean of each feature per group, used as reference values in the explanations. |

## Source corpus (not redistributed)

The features were computed from English discourse transcripts of
[AphasiaBank](https://aphasia.talkbank.org/) (MacWhinney et al., 2011):

- **Aphasic corpus** - Adler, Kurland, Neural, SCALE, Tucson, Williamson (289 transcripts)
- **Control corpus** - Baycrest, MSU, UMD, Wright (174 transcripts in the final feature file; 180 collected)

AphasiaBank is restricted to registered members and its transcripts may not be
redistributed, so the raw `.cha` files are **not** included here.
To rebuild the features from scratch, request access on the AphasiaBank website,
extract the participant utterances (interviewer turns removed), and apply
`carephasia.features.extract_features` to each transcript.
