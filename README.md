## TelME-SC : Self-Adaptive Context Teacher-leading Multi-modal Emotion Recognition for Real-time Online Inference

Abstract: Multimodal Emotion Recognition in Conversations (MERC) aims to predict human emotions through the combination and subsequent analysis of multiple modalities such as speech, video, and audio. Recent advances in the field have shown promise; however, much of the literature is focused only on offline inference, with static context windows. These approaches are not suitable for applications requiring real-time live inference, which is of high importance for a technology such as MERC. We propose TelME-SC, a novel architecture that combines the teacher-led attention shifting fusion architecture of TelME with the self-adaptive context modelling of SCMM. TelME- SC is fully capable of online inference operating on a per-utterance level. TelME-SC implements self-adaptive context by making use of both long and short-range context via its three complementary paths (global, local, and direct). To account for true real-time online inference, we propose a combination of Voice Activity Detection and Automatic Speech Recognition with TelME-SC to classify unsegmented, live, in-the-wild data. Experiments on the noisy, in-the-wild dataset AffWild-2 show a 1.58pp weighted F1- score improvement compared to the original TelME architecture on the same dataset, and a −0.53pp decrease on the standard, clean, benchmark dataset MELD. A live demo of the model pipeline shows a median delay of ∼ 1.5 − 1.7s. These results highlight that self-adaptive context is effective for MERC, but offers more benefits in noisy scenarios, and that online, utterance-based, multimodal emotion recognition is feasible without substantial accuracy loss.

# Repository notes

Within this file is several components of the dissertation's code:
    AffWild-2 train and model code
    VAD + ASR code
    MELD train and model code
    Live demo
Model checkpoints have not been included as they are too large, but can be supplied if requested.
Additionally, some trivial elements of the project have been omitted, such as CSV generation and filtering,
and other small scripts of that nature. I have also omitted all code used to train just TelME, only keeping
code that is relevant to TelME-SC, as all original TelME code can be found on the original repository.

The original TelME github can be found here: https://github.com/yuntaeyang/TelME
All train and model code for TelME components are based on the code provided from this repository. 
All SCMM components are coded from scratch.

