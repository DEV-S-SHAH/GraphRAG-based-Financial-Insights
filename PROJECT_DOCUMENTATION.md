# LTIMindtree Multi-Year GraphRAG System

## Project Overview

This is a sophisticated **Graph Retrieval-Augmented Generation (GraphRAG)** system for **LTIMindtree Limited** (formerly L&T Infotech merged with Mindtree), an Indian multinational IT services and consulting company. The system processes **four fiscal years** (FY2022-23 through FY2025-26) of annual reports to enable intelligent question-answering about the company's financial performance, strategic initiatives, risks, and entity relationships.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Data Pipeline](#data-pipeline)
4. [Core Components](#core-components)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Checkpoint Files](#checkpoint-files)
9. [Sample Outputs](#sample-outputs)
10. [Evaluation Results](#evaluation-results)
11. [Key Features](#key-features)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The system follows a **hybrid retrieval architecture** combining:

```
                    ┌─────────────────┐
                    │   User Question │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              Question Classification Router              │
│   (Financial / Graph / Narrative / Hybrid)               │
└─────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    Neo4j Graph Store     │    │   FAISS Vector Store    │
│  • Financial Facts      │    │  • Document Chunks       │
│  • Entities              │    │  • Semantic Search      │
│  • Relationships         │    │  • Page Provenance       │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Context Fusion Layer                     │
│   (Combines graph + vector evidence with fiscal year     │
│    awareness)                                             │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Ollama LLM (qwen2.5:3b)                     │
│   • Evidence-grounded answer generation                  │
│   • Explicit inference/evidence distinction               │
│   • Source attribution with page numbers                  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Structured Answer                       │
│   (Answer + Evidence + Sources + Metadata)               │
└─────────────────────────────────────────────────────────┘
```