<div align="center">

# ⚛️ Quantum-Inspired Prompt Optimization (Q-GAAPO)

### A Quantum-Enhanced Genetic Algorithm for Automatic Prompt Engineering on Hate Speech Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Powered-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Gemini API](https://img.shields.io/badge/Gemini-API-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

**This is a base/prototype implementation using smaller datasets for demonstration purposes.** The system uses the **ETHOS multilabel hate speech dataset** and **Google Gemini** as the LLM evaluator.

[📄 Base Paper](#-base-paper--inspiration) · [🏗️ Architecture](#%EF%B8%8F-system-architecture) · [🚀 Quick Start](#-quick-start) · [⚙️ Configuration](#%EF%B8%8F-configuration--customization)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Base Paper & Inspiration](#-base-paper--inspiration)
- [Key Innovation](#-key-innovation--novelty)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [Pipeline Deep-Dive](#-pipeline-deep-dive)
  - [Stage 1 — Quantum Sampling](#stage-1--quantum-sampling-quantum_samplerpy)
  - [Stage 2 — Chromosome Decoding](#stage-2--chromosome-decoding-decoderpy)
  - [Stage 3 — Prompt Construction](#stage-3--prompt-construction-prompt_builderpy)
  - [Stage 4 — LLM Evaluation](#stage-4--llm-evaluation-evaluatorpy)
  - [Stage 5 — Evolutionary Update](#stage-5--evolutionary-update--selection)
- [Gene Encoding Schema](#-gene-encoding-schema)
- [Evolution Loop](#-evolution-loop)
- [Dataset — ETHOS Multilabel](#-dataset--ethos-multilabel)
- [Quick Start](#-quick-start)
- [Configuration & Customization](#%EF%B8%8F-configuration--customization)
- [Project Structure](#-project-structure)
- [Results & Interpretation](#-results--interpretation)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)

---

## 🧬 Overview

**Q-GAAPO** (Quantum-Inspired Genetic Algorithm Applied to Prompt Optimization) is a prototype system that combines **quantum computing principles** with **genetic algorithm strategies** to automatically discover optimal prompts for Large Language Models (LLMs).

Instead of relying on manual prompt engineering or expensive LLM-based prompt rewriting (like APO or OPRO), this system:

1. **Encodes prompt structure as a binary chromosome** — each prompt configuration maps to an 11-bit string.
2. **Uses a Qiskit quantum circuit** to sample candidate chromosomes via parameterized rotation gates.
3. **Decodes bitstrings into structured prompts** with discrete genes (persona, reasoning style, etc.).
4. **Evaluates prompts** against real hate speech data using Google Gemini.
5. **Evolves rotation angles** toward elite candidates across generations.

> [!TIP]
> **Core Insight:** By converting prompt optimization from a free-text evolution problem into a *structured gene-based search problem*, we make it compatible with quantum-inspired search while drastically reducing the number of expensive LLM-based prompt rewrites.

---

## 📄 Base Paper & Inspiration

This project draws direct inspiration from the following published research:

> **GAAPO: Genetic Algorithm Applied to Prompt Optimization** > Xavier Sécheresse, Jacques-Yves Guilbert–Ly, Antoine Villedieu de Torcy  
> *Frontiers in Artificial Intelligence*, Volume 8, 2025  
> DOI: [10.3389/frai.2025.1613007](https://doi.org/10.3389/frai.2025.1613007)

### What the Paper Proposes
The GAAPO paper introduces a **hybrid genetic optimization framework** that evolves LLM prompts through successive generations. It integrates strategies like OPRO, APO, and random mutation within a single cycle:

| Strategy | Description |
| :--- | :--- |
| **OPRO** | Trajectory-based optimization using prompt history |
| **APO/ProTeGi** | Error-gradient-driven prompt refinement |
| **Random Mutator** | 8 distinct mutation strategies (persona injection, etc.) |
| **Crossover** | Split-and-merge recombination of parent prompts |

### How This Project Extends It
Our prototype takes GAAPO's core principle and adds a **quantum-inspired sampling layer**:
- **GAAPO** uses LLM-based generators (expensive API calls).
- **Q-GAAPO** uses a **Qiskit quantum circuit** to sample prompt structures (cheap, no LLM calls for generation).

---

## 🔬 Key Innovation & Novelty

```text
┌─────────────────────────────────────────────────────────────────┐
│                     TRADITIONAL APPROACH                        │
│  Free-text prompt → LLM rewrites → LLM evaluates → repeat       │
│  ($$$ expensive: LLM calls for BOTH generation AND evaluation)  │
└─────────────────────────────────────────────────────────────────┘
                        ↓ vs ↓
┌─────────────────────────────────────────────────────────────────┐
│                     OUR APPROACH (Q-GAAPO)                      │
│  Quantum circuit → Bitstring → Decoder → Structured Prompt      │
│  → LLM evaluates only → θ update → repeat                       │
│  ($ cheap: LLM calls ONLY for evaluation)                       │
└─────────────────────────────────────────────────────────────────┘