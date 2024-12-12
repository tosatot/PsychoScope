import os

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generator.generate import generator
from generator.model_analysis.model_analyzer import ModelAnalyzer, AnalysisConfig, AnalysisType
from generator.utils import get_questionnaire, get_persona, generate_testfile

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class QuestionnaireEnum(str, Enum):
    BFI = "BFI"
    DTDD = "DTDD"
    EPQ_R = "EPQ-R"
    ECR_R = "ECR-R" 
    CABIN = "CABIN"
    GSE = "GSE"
    LMS = "LMS"
    BSRI = "BSRI"
    ICB = "ICB"
    LOT_R = "LOT-R"
    EMPATHY = "Empathy"
    EIS = "EIS"
    WLEIS = "WLEIS"
    SIXTEEN_P = "16P"

class PersonaEnum(str, Enum):
    ASSISTANT = "assistant"
    BUDDHIST = "buddhist"
    DEPRESSION = "depression"
    SCHIZOPHRENIA = "schizophrenia"
    ANTISOCIAL = "antisocial"
    ANXIETY = "anxiety"
    BIPOLAR = "bipolar"
    BORDERLINE = "borderline"
    NARCISSISTIC = "narcissistic"
    AUTISM = "autism"
    PSYCHOTHERAPIST = "psychotherapist"
    TEACHER = "teacher"

class ModelTestConfig(BaseModel):
    api_key: str
    model_name: str
    questionnaire: QuestionnaireEnum
    persona: PersonaEnum
    environment: str
    batch_size: int = Field(default=1, ge=1, le=10)
    use_conversation_history: bool = Field(default=False)
    test_count: int = Field(default=1, ge=1)
    variability_source: str = Field(default="shuffle")
    num_iterations: int = Field(default=1, ge=1)

class AnalysisRequest(BaseModel):
    analysis_types: List[str]
    input_file: str
    output_dir: str
    questionnaire_name: Optional[str]
    formula: Optional[str] = None
    model_type: Optional[str] = None
    distribution: Optional[str] = None
    link_function: Optional[str] = None

class JobStatusResponse(BaseModel):
    status: JobStatus
    message: str = ""
    progress: float = 0.0

job_statuses: Dict[str, JobStatusResponse] = {}

app = FastAPI(
    title="PsychoBench API",
    description="API for psychological assessment of language models",
    version="1.0.0"
)

@app.post("/test/")
async def test_model(config: ModelTestConfig, background_tasks: BackgroundTasks):
    try:
        # Create output directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/figures", exist_ok=True)

        # Generate unique job ID
        job_id = f"{config.model_name}_{config.questionnaire}_{config.persona}_{config.variability_source}"

        # Initialize job status
        job_statuses[job_id] = JobStatusResponse(
            status=JobStatus.PENDING,
            message="Job initialized"
        )

        # Configure test parameters
        test_args = Args(**{
            'API_KEY': config.api_key,
            'model': config.model_name,
            'b_size': config.batch_size,
            'conv_hist': config.use_conversation_history,
            'test_count': config.test_count,
            'n_iter': config.num_iterations,
            'testing_file': f'results/{job_id}.csv',
            'variab_source': config.variability_source,
            'environment': config.environment
        })

        # Get questionnaire and persona configurations
        try:
            questionnaire = get_questionnaire(config.questionnaire)
            persona = get_persona(config.persona)
        except Exception as e:
            job_statuses[job_id].status = JobStatus.FAILED
            job_statuses[job_id].message = str(e)
            raise HTTPException(status_code=500, detail=str(e))

        # Update status to running
        job_statuses[job_id].status = JobStatus.RUNNING
        job_statuses[job_id].message = "Processing test data"

        # Add background task with status updates
        async def run_test_with_status():
            try:
                # Generate test file
                generate_testfile(questionnaire, test_args, config.variability_source)

                # Run generator
                generator(questionnaire, persona, test_args, config.variability_source)

                # Update status to complete
                job_statuses[job_id].status = JobStatus.COMPLETE
                job_statuses[job_id].message = "Test completed successfully"
                job_statuses[job_id].progress = 1.0

            except Exception as e:
                job_statuses[job_id].status = JobStatus.FAILED
                job_statuses[job_id].message = str(e)
                raise

        background_tasks.add_task(run_test_with_status)

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Test started. Use job_id to check status and retrieve results."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/")
async def analyze_results(request: AnalysisRequest):
    """
    Run analyses on test results. Supports:
    - Exploratory data analysis
    - Log-log mean plots
    - Log-log variance plots
    - Radar plots
    - Violin plots
    - Regression analysis
    - Repeated measures analysis
    - ANOVA
    """
    try:
        analysis_types = []
        for analysis in request.analysis_types:
            try:
                analysis_types.append(AnalysisType[analysis.upper()])
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analysis type: {analysis}. Valid types are: {[t.name for t in AnalysisType]}"
                )

        # Create analysis configuration
        config = AnalysisConfig(
            input_file=request.input_file,
            output_dir=request.output_dir,
            questionnaire_name=request.questionnaire_name,
            analysis_types=analysis_types,
            formula=request.formula,
            model_type=request.model_type,
            distribution=request.distribution,
            link_function=request.link_function
        )

        # Initialize analyzer and run analyses
        analyzer = ModelAnalyzer(config)
        analyzer.run_all_analyses()

        return {
            "status": "complete",
            "output_dir": request.output_dir,
            "message": "Analysis complete. Results available in output directory."
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions with their original status code
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a model testing job"""
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_statuses[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    try:
        result_file = f'results/{job_id}.csv'
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail="Results not found. Job may still be running or failed."
            )

        # Read and return results
        try:
            results = pd.read_csv(result_file)
            return {
                "status": "success",
                "results": results.to_dict(orient='records')
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading results: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Retrieve results for a completed job
    """
    try:
        result_file = f'results/{job_id}.csv'
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail="Results not found. Job may still be running or failed."
            )

        # Read and return results
        results = pd.read_csv(result_file)
        return {
            "status": "success",
            "results": results.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{job_id}")
async def list_job_files(job_id: str):
    """List all available files for a given job ID"""
    try:
        # Define possible directories where files might be stored
        base_dirs = {
            "results": "data",
            "results/figures": "figure",
            f"results/prompts_{job_id}": "prompt",
            f"results/responses_{job_id}": "response"
        }

        available_files = []

        for directory, file_type in base_dirs.items():
            if os.path.exists(directory):
                # List all files in directory
                files = os.listdir(directory)
                for file in files:
                    if not os.path.isfile(os.path.join(directory, file)):
                        continue

                    file_path = os.path.join(directory, file)
                    file_size = os.path.getsize(file_path)

                    # For prompt and response directories, include all files
                    # For other directories, only include files containing job_id
                    if ("prompts_" in directory or "responses_" in directory) or job_id in file:
                        available_files.append({
                            "filename": file,
                            "path": file_path,
                            "size": file_size,
                            "type": file_type
                        })

        if not available_files:
            raise HTTPException(
                status_code=404, 
                detail=f"No files found for job ID: {job_id}"
            )

        return {
            "job_id": job_id,
            "files": available_files
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error listing files: {str(e)}"
        )

# Add download endpoint
@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a specific file for a given job ID"""
    try:
        # Check possible locations for the file
        possible_paths = [
            os.path.join("results", filename),
            os.path.join("results", "figures", filename),
            os.path.join("results", f"prompts_{job_id}", filename),
            os.path.join("results", f"responses_{job_id}", filename)
        ]

        # Find the first existing path
        file_path = next(
            (path for path in possible_paths if os.path.exists(path)), 
            None
        )

        if not file_path:
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}"
            )

        if not os.path.isfile(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"Path exists but is not a file: {filename}"
            )

        # Return file response with appropriate media type
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading file: {str(e)}"
        )
