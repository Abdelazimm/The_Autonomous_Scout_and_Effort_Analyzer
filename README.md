# The Autonomous Scout & Effort Analyzer

An AI-powered soccer scouting system that fuses computer vision (object detection, tracking, movement metrics) with audio transcription (commentary analysis) to generate automated scouting and effort reports.

## Phase 1 — Repository Scaffolding

This phase establishes the project structure, dependency management, CI pipeline, shared schemas, and data loader wrappers.

### Project Structure

```
The Autonomous Scout & Effort Analyzer/
├── vision/              # Detection, tracking, movement metrics, schemas
├── audio/               # Transcription, timestamp linker
├── fusion/              # Aligner, context builder, orchestrator
├── report/              # Report generator + templates
├── api/                 # FastAPI production API layer (for React frontend)
├── frontend/            # React frontend (future phases)
├── tools/                # Gradio tester and dev utilities
├── data/                # Data loaders + raw datasets
│   ├── roboflow/
│   ├── soccertrack/
│   └── custom_clips/
├── tests/               # Unit tests
├── notebooks/           # Experimental notebooks
├── configs/             # Pipeline configuration (pipeline.yaml)
└── .github/workflows/   # CI (ruff lint + pytest)
```

### Unified Taxonomy

Adopted from the 4 Roboflow classes:

| ID | Class      |
|----|------------|
| 0  | ball       |
| 1  | player     |
| 2  | referee    |
| 3  | goalkeeper |

### Setup

```bash
pip install -r requirements.txt
pytest tests/ -v
ruff check .
```
