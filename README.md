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

## Results

### Model Performance
- **HeavyBERTa**: 89.12% accuracy in predicting missing residues
- **LightGPT**: 86.35% accuracy in light chain prediction
- **B cell classification**: 92.31% accuracy for heavy chains, 79.17% for light chains

### Generated Sequences
- High germline similarity (93.68% for conditioned generation)
- Proper structural preservation with pTM scores 0.94-0.95
- Enhanced sequence completeness compared to unconditioned generation
- Chain-type specific patterns with distinct distributions for κ and λ chains

### Biological Validation
- Confirmed V gene coherence patterns in memory vs. naive B cells
- Generated sequences maintain essential structural constraints
- ImmunoMatch pairing compatibility scores validate biological relevance

## Technical Implementation

### Models
- **HeavyBERTa**: 4-12 layer RoBERTa configurations (13M-86M parameters)
- **LightGPT**: 12-layer GPT-2 architecture (86M parameters)
- **Heavy2Light**: Encoder-decoder with cross-attention mechanisms

### Training
- Pre-training on NVIDIA A100/H100 GPUs with 80GB RAM
- HuggingFace Transformers and Adapters libraries
- Parameter-efficient fine-tuning for translation tasks

### Data Processing
- Hierarchical clustering for robust train/test splits
- Quality filtering for healthy donor sequences
- Integration of OAS and PLAbDab databases

## Future Directions

- Enhanced pairing accuracy through diverse datasets
- Integration of structural constraints during training
- Alternative generative architectures (diffusion models)
- Functional validation (antigen specificity, neutralization)
- Application to disease and vaccination studies

## Significance

This work demonstrates the feasibility of using deep learning to model complex sequence relationships in antibody repertoires. By combining discriminative and generative modeling approaches, the framework reveals immunological insights and enables context-aware design of antibody sequences, setting the stage for data-driven advances in immunotherapy, vaccine design, and synthetic immunology.
