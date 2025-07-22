# A Deep Learning Framework for Heavy-to-Light Antibody Sequence Translation

## Background

Antibodies are critical effector molecules of the adaptive immune system, consisting of two identical heavy chains (HCs) and two identical light chains (LCs) arranged in a characteristic Y-shaped configuration. The immense diversity of antibodies arises from precise genetic mechanisms including V(D)J recombination, junctional diversity, and somatic hypermutation. Understanding how this diversity is organized and utilized within antibody repertoires is essential for advancing fundamental immunology, rational vaccine design, and the development of engineered therapeutic antibodies.

Next-generation sequencing (NGS) has revolutionized antibody repertoire studies by enabling high-throughput characterization of immune responses at unprecedented scale. However, NGS typically loses the natural pairing information between heavy and light chains during sequencing, creating a critical knowledge gap since antibody function and specificity critically depend on the correct association of HCs and LCs.

## Problem Statement

A key challenge in computational immunology is accurately reconstructing or predicting heavy-light chain pairing relationships from sequence data alone. This is complicated by:

- **Scale and complexity**: Antibody repertoire datasets contain massive amounts of sequence data
- **Lost pairing information**: NGS loses natural HC-LC associations during sequencing
- **Functional dependency**: Antibody specificity critically depends on correct HC-LC pairing
- **Limited paired data**: Public databases contain billions of unpaired sequences but only millions of paired examples

## Approach

This work implements a novel two-stage deep learning strategy:

### Stage 1: Pre-training Domain-Specific Language Models
- **HeavyBERTa**: RoBERTa-based masked language model for heavy chain sequences
- **LightGPT**: GPT-2-based causal language model for light chain sequences
- Trained on >99 million HC and >22 million LC sequences from healthy human donors

### Stage 2: Translation Model Development
- **Heavy2Light**: Encoder-decoder architecture combining pre-trained models
- Uses parameter-efficient adapter-based fine-tuning
- Translates heavy chain sequences into corresponding light chains
- Trained on 470k paired sequences from OAS and PLAbDab databases

## Getting started
### Installation  
Environment set up to train the models with adapters using conda and [adapter_env.txt](environments/adapter_env.txt)  
```
conda create --name adapter_env --file adapter_env.txt
```
