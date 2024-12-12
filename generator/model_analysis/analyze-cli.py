#!/usr/bin/env python3
import argparse
import sys
import os
from typing import List, Optional
from pathlib import Path
import json
from model_analyzer import ModelAnalyzer, AnalysisConfig, AnalysisType

class Config:
    """Configuration handler for the analysis tool"""
    def __init__(self):
        # Get the directory where the script is located
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Get project root (parent directory of the script directory)
        self.project_root = self.script_dir.parent
        
        # Look for config in standard locations, starting with project root
        config_locations = [
            self.project_root / 'questionnaires.json',          # Project root
            self.project_root / 'config' / 'questionnaires.json', # Project config dir
            self.script_dir / 'questionnaires.json'              # Script directory
        ]
        
        self.questionnaire_file = None
        for loc in config_locations:
            if loc.is_file():
                self.questionnaire_file = loc
                print(f"Found questionnaires.json at: {loc}")  # Debug info
                break
        
        if not self.questionnaire_file:
            raise FileNotFoundError(
                "questionnaires.json not found in any of:\n" +
                "\n".join(str(loc) for loc in config_locations)
            )
    
    def get_questionnaire_path(self) -> Path:
        """Return path to questionnaire file"""
        return self.questionnaire_file
    
    def load_questionnaire(self, questionnaire_name: Optional[str] = None) -> dict:
        """Load and return questionnaire data"""
        try:
            with open(self.questionnaire_file, 'r') as f:
                data = json.load(f)
                
            if questionnaire_name:
                # If name specified, return only that questionnaire
                for questionnaire in data['questionnaires']:
                    if questionnaire['name'].lower() == questionnaire_name.lower():
                        return questionnaire
                raise ValueError(f"Questionnaire '{questionnaire_name}' not found in {self.questionnaire_file}")
            
            return data
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing questionnaire file {self.questionnaire_file}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading questionnaire file {self.questionnaire_file}: {str(e)}")


def parse_analysis_types(analysis_types: str) -> List[AnalysisType]:
    """Convert comma-separated string of analysis types to list of AnalysisType enums"""
    if 'run_all' in analysis_types:
        return [type_ for name, type_ in AnalysisType.__members__.items()]
        
    valid_types = {name.lower(): type_ for name, type_ in AnalysisType.__members__.items()}
    requested_types = [t.strip().lower() for t in analysis_types.split(',')]
    
    analysis_list = []
    for type_name in requested_types:
        if type_name not in valid_types:
            print(f"Warning: Invalid analysis type '{type_name}'. Skipping.")
            continue
        analysis_list.append(valid_types[type_name])
    
    return analysis_list

def validate_args(args: argparse.Namespace, config: Config) -> bool:
    """Validate command line arguments based on requested analyses"""
    if not args.analysis_types:
        print("Error: No analysis types specified. Use --analysis-types to specify analyses.")
        return False
        
    analysis_types = parse_analysis_types(args.analysis_types)
    
    # Special handling for prepare_data
    if AnalysisType.PREPARE_DATA in analysis_types:
        if not args.input_dir:
            print("Error: Input directory is required for data preparation.")
            return False
            
        # Validate input directory exists
        input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
        if not os.path.isdir(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            return False
            
        # Try to load questionnaire to validate it exists and is readable
        try:
            config.load_questionnaire(args.questionnaire_name)
        except Exception as e:
            print(f"Error with questionnaire configuration: {str(e)}")
            return False
    else:
        # For other analyses, need formatted input file
        if not args.input_file:
            print("Error: Input file is required for analyses other than data preparation.")
            return False
            
    if AnalysisType.REGRESSION in analysis_types:
        if not args.formula:
            print("Error: Formula is required for regression analysis.")
            return False
        if not args.model_type:
            print("Error: Model type is required for regression analysis.")
            return False
    
    # Validate output directory
    try:
        output_dir = os.path.abspath(os.path.expanduser(args.output))
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return False
    
    return True

def setup_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="""
        Model Analysis Tool - Run various analyses on model performance data.
        
        Example usage:
        # Prepare data:
        analyze.py --input-dir raw_data --output results --analysis-types prepare_data --questionnaire-name BFI
        
        # Run analyses on prepared data:
        analyze.py --input-file results/prepared_data.csv --output results --analysis-types loglog_mean,radar

        # Run all steps:
        analyze.py --input-dir raw_dir --input-file results/prepared_data.csv --output results --analysis-types run_all --questionnaire-name BFI --formula "score ~ persona * model_type"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input arguments (either one or the other or both but can't be none; required with validate_args) 
    parser.add_argument(
        '--input-dir',
        help='Directory containing raw CSV files (for data preparation)'
    )
    parser.add_argument(
        '--input-file',
        help='Path to formatted input CSV file (for analysis)'
    )
    
    # Required arguments
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '-a', '--analysis-types',
        required=True,
        help="""
        Comma-separated list of analyses to run. Available types:
        prepare_data: Prepare and clean raw data files
        exploratory: Run exploratory data analysis
        loglog_mean: Generate log-log mean plots
        loglog_variance: Generate log-log variance plots
        radar: Generate radar plots
        violin: Generate violin plots
        regression: Run regression analysis
        repeated_measures: Run repeated measures analysis
        anova: Run ANOVA analysis
        run_all: Run all steps including prepare_data
        """
    )
    
    # Optional arguments
    parser.add_argument(
        '-n', '--questionnaire-name',
        help='Name of questionnaire (e.g., BFI) for data preparation'
    )
    
    parser.add_argument(
        '-f', '--formula',
        help='Formula for regression analysis (required for regression)'
    )
    
    parser.add_argument(
        '-m', '--model-type',
        choices=['anova', 'glm', 'logit', 'bayes'],
        help='Model type for regression analysis (required for regression)'
    )
    
    parser.add_argument(
        '-d', '--distribution',
        choices=['Gamma', 'Gaussian', 'Binomial'],
        default='Binomial',
        help='Distribution family for GLM regression (default: Binomial)'
    )
    
    parser.add_argument(
        '-l', '--link-function',
        choices=['Log', 'Logit', 'Probit'],
        default='Logit',
        help='Link function for GLM regression (default: Logit)'
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging verbosity level (default: INFO)'
    )
    
    return parser

def main():
    # Initialize configuration
    try:
        config = Config()
        questionnaire_path = str(config.get_questionnaire_path())
    except Exception as e:
        print(f"Error initializing configuration: {str(e)}")
        sys.exit(1)
    
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args, config):
        parser.print_help()
        sys.exit(1)
    
    try:
        analysis_types = parse_analysis_types(args.analysis_types)
        
        # Handle data preparation
        if AnalysisType.PREPARE_DATA in analysis_types:
            # Convert paths to absolute paths
            input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
            output_dir = os.path.abspath(os.path.expanduser(args.output))
            
            config = AnalysisConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                questionnaire_file=questionnaire_path,
                questionnaire_name=args.questionnaire_name,
                analysis_types=[AnalysisType.PREPARE_DATA],
                verbosity=args.verbosity
            )
            
            analyzer = ModelAnalyzer(config)
            analyzer.prepare_data()
            prepared_file = os.path.join(output_dir, f"{args.questionnaire_name}_data.csv")
            print(f"Data preparation complete. Formatted data saved to: {prepared_file}")
            
            # Remove prepare_data from analysis types for subsequent operations
            analysis_types.remove(AnalysisType.PREPARE_DATA)

        # Run other analyses if requested
        if analysis_types:
            input_file = args.input_file or os.path.join(args.output, f"{args.questionnaire_name}_data.csv")
            input_file = os.path.abspath(os.path.expanduser(input_file))
            output_dir = os.path.abspath(os.path.expanduser(args.output))
            
            config = AnalysisConfig(
                input_file=input_file,
                output_dir=output_dir,
                questionnaire_file=questionnaire_path,
                questionnaire_name=args.questionnaire_name,
                analysis_types=analysis_types,
                formula=args.formula,
                model_type=args.model_type,
                distribution=args.distribution,
                link_function=args.link_function,
                verbosity=args.verbosity
            )
            
            analyzer = ModelAnalyzer(config)
            analyzer.run_all_analyses()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    print("Analysis completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
