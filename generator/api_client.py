import requests
from typing import Optional, Dict, List, Union
import os
import time
from enum import Enum
import json
from pathlib import Path

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

class PsychoBenchClient:
    def __init__(self, base_url: str):
        """Initialize the PsychoBench API client
        
        Args:
            base_url (str): Base URL of the PsychoBench API
        """
        self.base_url = base_url.rstrip('/')
        
    def submit_test(self, config: Dict) -> str:
        """Submit a new model test job
        
        Args:
            config (Dict): Test configuration dictionary
            
        Returns:
            str: Job ID for the submitted test
        """
        response = requests.post(f"{self.base_url}/test/", json=config)
        response.raise_for_status()
        return response.json()["job_id"]
    
    def get_status(self, job_id: str) -> Dict:
        """Get the status of a submitted job
        
        Args:
            job_id (str): Job ID to check
            
        Returns:
            Dict: Job status information
        """
        response = requests.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_results(self, job_id: str) -> Dict:
        """Get results for a completed job
        
        Args:
            job_id (str): Job ID to get results for
            
        Returns:
            Dict: Job results
        """
        response = requests.get(f"{self.base_url}/results/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def list_files(self, job_id: str) -> Dict:
        """List available files for a job
        
        Args:
            job_id (str): Job ID to list files for
            
        Returns:
            Dict: Dictionary containing file information
        """
        response = requests.get(f"{self.base_url}/files/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def download_file(self, job_id: str, filename: str, output_dir: Optional[str] = None) -> str:
        """Download a specific file from a job
        
        Args:
            job_id (str): Job ID the file belongs to
            filename (str): Name of the file to download
            output_dir (str, optional): Directory to save the file to. Defaults to current directory.
            
        Returns:
            str: Path to the downloaded file
        """
        response = requests.get(
            f"{self.base_url}/download/{job_id}/{filename}", 
            stream=True
        )
        response.raise_for_status()
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = filename
            
        # Write file in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        return output_path
    
    def download_all_files(self, job_id: str, output_dir: Optional[str] = None) -> List[str]:
        """Download all available files for a job
        
        Args:
            job_id (str): Job ID to download files for
            output_dir (str, optional): Directory to save files to. Defaults to current directory.
            
        Returns:
            List[str]: List of paths to downloaded files
        """
        # Get list of available files
        files_info = self.list_files(job_id)
        downloaded_files = []
        
        # Download each file
        for file_info in files_info["files"]:
            filename = file_info["filename"]
            try:
                file_path = self.download_file(job_id, filename, output_dir)
                downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                
        return downloaded_files
    
    def wait_for_completion(
        self, 
        job_id: str, 
        polling_interval: int = 10,
        timeout: Optional[int] = None
    ) -> Dict:
        """Wait for a job to complete
        
        Args:
            job_id (str): Job ID to wait for
            polling_interval (int): Seconds between status checks
            timeout (int, optional): Maximum seconds to wait
            
        Returns:
            Dict: Final job status
        """
        start_time = time.time()
        while True:
            status = self.get_status(job_id)
            
            if status["status"] in [JobStatus.COMPLETE, JobStatus.FAILED]:
                return status
                
            if timeout and (time.time() - start_time > timeout):
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
                
            time.sleep(polling_interval)

# Example usage:
if __name__ == "__main__":
    # Initialize client
    client = PsychoBenchClient("http://localhost:8000")
    
    # Submit a test
    config = {
        "api_key": "test-key",
        "model_name": "test-model",
        "questionnaire": "BFI",
        "persona": "assistant",
        "environment": "API",
        "batch_size": 1,
        "test_count": 1
    }
    
    try:
        # Submit job
        job_id = client.submit_test(config)
        print(f"Submitted job: {job_id}")
        
        # Wait for completion
        status = client.wait_for_completion(job_id, timeout=3600)
        print(f"Job completed with status: {status}")
        
        if status["status"] == JobStatus.COMPLETE:
            # Download results
            output_dir = f"downloaded_results_{job_id}"
            downloaded_files = client.download_all_files(job_id, output_dir)
            print(f"Downloaded {len(downloaded_files)} files to {output_dir}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
