### TODO:
- [ ] integrate the repo into astroid
- [ ] change config files to json
- [ ] fix hard coded scp files, and add separete dataset root in config
- [ ] separate output folder for each config
- [ ] implement debug mode in code, not from hardcoded separate dataset path

## paper link: https://arxiv.org/abs/2011.12022
### This is the repository for multi-decoder DPRNN, submitted to ICASSP 2021. It deals with source separation with variable number of speakers. It has 98.5% accuracy in speaker number of selection and similar SNR as models trained individually on fixed number of speakers, which is much higher than all previous SOTA methods.

### We are still refactoring the codebase, and will provide another useful update soon. Demo can be found at: https://junzhejosephzhu.github.io/Multi-Decoder-DPRNN/

Results
2    | 3    | 4    |5
-----|------|------|--
2998 | 17   | 1    |0
2    | 2977 | 27   |0
0    | 6    | 2928 |80
0    | 0    | 44   |2920