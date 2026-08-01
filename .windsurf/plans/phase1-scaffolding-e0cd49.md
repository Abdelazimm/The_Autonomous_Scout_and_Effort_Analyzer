# Phase 1: Repository Scaffolding

Set up the complete directory structure, dependency management, CI, config, data loaders, and unit tests for "The Autonomous Scout & Effort Analyzer," with a clear split between the testing UI (Gradio) and production API layer (FastAPI + future React frontend).

## Decisions
- **Dependency file**: `requirements.txt`
- **Linter**: `ruff`
- **Package structure**: `__init__.py` in every Python package directory
- **Stub files**: Each stub `.py` file gets a module-level docstring describing its purpose for Phase 2+

## Steps

### 1. Create directory structure
Create all directories per the spec:
- `vision/`, `audio/`, `fusion/`, `report/templates/`, `api/`, `frontend/`, `tools/`, `data/roboflow/`, `data/soccertrack/`, `data/custom_clips/`, `tests/`, `notebooks/`, `configs/`, `.github/workflows/`
- `frontend/` is left empty (React app in later phases)

### 2. Create stub Python files with docstrings
Each gets a one-line module docstring:
- `vision/detection.py`, `vision/tracking.py`, `vision/movement_metrics.py`
- `audio/transcription.py`, `audio/timestamp_linker.py`
- `fusion/aligner.py`, `fusion/context_builder.py`, `fusion/orchestrator.py`
- `report/generator.py`
- `api/main.py` (FastAPI app stub for production API layer)
- `tools/gradio_tester.py` (Gradio app for Colab/local testing & visual debugging)

### 3. Create `__init__.py` files
Add empty `__init__.py` to: `vision/`, `audio/`, `fusion/`, `report/`, `api/`, `data/`, `tests/`

### 4. Create `requirements.txt`
Pin: `ultralytics`, `opencv-python`, `gradio`, `pydantic`
Commented-out placeholders: `openai-whisper`, `langchain`, `bytetrack`, `fastapi`, `uvicorn`

### 5. Create `configs/pipeline.yaml`
Stub config with `class_map` section:
```yaml
class_map:
  0: ball
  1: player
  2: referee
  3: goalkeeper
```
Plus placeholder sections for vision, audio, fusion, report pipeline stages.

### 6. Create CI workflow `.github/workflows/ci.yml`
- Trigger: on push
- Steps: checkout, setup Python, install deps, run `ruff check`, run `pytest`

### 7. Implement `vision/schemas.py`
Pydantic models:
- `Detection`: frame_idx, class_id, class_name, bbox (x, y, w, h), confidence
- `GroundTruthFrame`: frame_idx, timestamp, detections list

### 8. Implement `data/roboflow_loader.py`
Function `load_roboflow_yolo(label_path, class_map)` → parses YOLO `.txt` label file, returns `GroundTruthFrame`.

### 9. Implement `data/soccertrack_loader.py`
Function `load_soccertrack_frame(match_data, frame_idx)` → wraps SoccerTrack's `load_match()` output, returns `GroundTruthFrame`.

### 10. Write unit tests
- `tests/test_roboflow_loader.py`: mock YOLO label file, assert correct `Detection` parsing
- `tests/test_soccertrack_loader.py`: mock match data, assert correct `GroundTruthFrame`
- `tests/test_movement_metrics.py`: empty placeholder (Phase 2)
- `tests/test_aligner.py`: empty placeholder (Phase 2)
- `tests/test_fusion_prompt.py`: empty placeholder (Phase 2)

### 11. Create `.gitignore`
Standard Python `.gitignore` (venv, __pycache__, .env, data files, etc.)

### 12. Create `README.md`
Brief project description and Phase 1 scope.
