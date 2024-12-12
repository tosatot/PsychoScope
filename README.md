# PsychoScope: Psychological Assessment Framework for LLMs


<p align="center">
  <img src="psychoscope.png" width="400" alt="Psychoscope">
</p>

PsychoScope is a Python framework for conducting systematic psychological assessments of Large Language Models (LLMs). It provides a robust suite of tools for administering standardized psychological questionnaires, collecting responses, and performing detailed statistical analyses of model behavior across different personality frameworks and conditions. Notably, these includes analyses on the stability of the mesured personality trait scores.



This research has been published in:

- Evaluating Evaluations Workshop @NeuroIPS 2024
https://evaleval.github.io/accepted_papers/EvalEval_24_Tosato.pdf

- Behavioural Machine Learning Workshop @NeuroIPS 2024
https://openreview.net/forum?id=vBg3OvsHwv


## Key Features

### Psychological Assessment
- Multiple standardized questionnaires support:
  - Big Five Inventory (BFI)
  - Eysenck Personality Questionnaire Revised (EPQ-R)
  - Short Dark Triad  (SDT)
  - Additional assessments

### Model Evaluation
- Support for several LLM families.
- Batch and sequential testing modes
- Optional inclusion of Conversation History
- Persona-based evaluation across:
  - Standard assistant role
  - Clinical conditions (e.g., depression, anxiety, etc.)
  - Professional roles (e.g., teacher, psychotherapist)
  - Cultural perspectives (e.g., Buddhist monk)

### Analysis Tools
- Statistical analysis suite:
  - Trait stability assessment
  - Variance scaling analysis
  - Mixed-effects regression models
  - ANOVA and post-hoc testing
  
- Visualization toolkit:
  - Model scaling behavior plots
  - Response distribution analysis
  - Trait variance analysis
  - Persona comparison radar plots



## Usage

### Installation and Setup

```bash
# Create virtual environment
python3 -m venv psycho-env
source psycho-env/bin/activate  # Windows: .\psycho-env\Scripts\activate

# Install requirements and package
pip install -r requirements.txt
pip install -e .
```


### Command Line Interface

Basic usage pattern:
```bash
python analyze-cli.py --input-dir <data_dir> \
                     --output <output_dir> \
                     --analysis-types <types> \
                     --questionnaire-name <name>
```

#### Key Arguments
- `--input-dir`: Raw data directory
- `--output`: Results directory
- `--analysis-types`: Analysis types (comma-separated)
- `--questionnaire-name`: Target questionnaire

Available analyses: `prepare_data`, `loglog_mean`, `loglog_variance`, `radar`, `violin`, `regression`, `repeated_measures`, `anova`, `run_all`

### Quick Start Examples

1. Data Preparation:
```bash
python analyze-cli.py --input-dir raw_data --output results \
                     --analysis-types prepare_data --questionnaire-name BFI
```

2. Visualization Analysis:
```bash
python analyze-cli.py --input-file results/BFI_data.csv \
                     --output results --analysis-types loglog_mean,radar
```

### Python API

```python
from model_analyzer import ModelAnalyzer, AnalysisConfig, AnalysisType

# Basic analysis
config = AnalysisConfig(
    input_dir="raw_data",
    output_dir="results",
    questionnaire_name="BFI",
    analysis_types=[AnalysisType.PREPARE_DATA, AnalysisType.LOGLOG_MEAN]
)

analyzer = ModelAnalyzer(config)
analyzer.run_all_analyses()
```

## Data Format and Configuration

### Input Data
CSV files with columns:
- `model`: Model identifier
- `persona`: Persona identifier
- `trait`: Trait being measured
- `score`: Numerical score

### Configuration
- `questionnaires.json` defines test parameters and scoring
- Searched in: project root, config directory, script directory
- Logs stored in: `prepare_data.log`, `plot_violins.log`

## Testing

```bash
# Run test suite
pytest tests/

# With coverage
pytest tests/ --cov=generator

# Specific tests
pytest tests/test_model_analyzer.py
pytest -k "test_prepare_data"
```



## Contributors
These framework was the results of collaborative work. Main contributors are:

- Mahmood Hegazy 
- David Lemay
- Mohammed Abukalam
- Tommaso Tosato
- Guillaume Dumas
- Irina Rish


## Attribution and License

This project contains code derived from https://github.com/CUHK-ARISE/PsychoBench. Key components that have been adapted include:
- Data structures and processing pipeline
- Basic questionnaire handling system


Major enhancements and modifications in this project include:
- Enhanced analysis capabilities and statistical methods
- Visualization options and plotting functions
- Extended questionnaires availability
- Modified and optimized data processing pipeline
- New command-line interface and analysis workflow
- Tools for analysing stability of personality traits.
- API

This project and the original Psychobench are licensed under GPL-3.0. The full license text can be found in the LICENSE file.