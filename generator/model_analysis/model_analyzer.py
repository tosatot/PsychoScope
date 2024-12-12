from typing import List, Optional, Union
import json
import pandas as pd
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto


# Import modules using relative imports
from generator.model_analysis.exploratory_data_analysis import run_eda
from generator.model_analysis.prepare_data import prepare_data as prep_data
from generator.model_analysis.plot_loglog_mean import plot_scaling_behavior, load_human_baselines as load_mean_human_baselines, prepare_data as prepare_mean_data
from generator.model_analysis.plot_loglog_var import plot_variance_scaling, load_human_baselines as load_var_human_baselines, prepare_data as prepare_var_data
from generator.model_analysis.plot_radar import create_faceted_radar_plot
from generator.model_analysis.plot_violins import plot_data as plot_violin_data
from generator.model_analysis.stat_regression import run_regression_analysis
from generator.model_analysis.stat_repeated_measures_mixed_lm import main as rm_main
from generator.model_analysis.repeated_measures_anova import main as anova_main
from generator.utils import get_questionnaire

class AnalysisType(Enum):
    EXPLORATORY = auto()
    PREPARE_DATA = auto()
    LOGLOG_MEAN = auto()
    LOGLOG_VARIANCE = auto()
    RADAR = auto()
    VIOLIN = auto()
    REGRESSION = auto()
    REPEATED_MEASURES = auto()
    ANOVA = auto()
    RUN_ALL = auto()

@dataclass
class AnalysisConfig:
    output_dir: str
    analysis_types: List[AnalysisType] = None
    # For data preparation
    input_dir: Optional[str] = None
    questionnaire_file: Optional[str] = None
    questionnaire_name: Optional[str] = None
    # For analysis functions
    input_file: Optional[str] = None
    # For regression analysis
    formula: Optional[str] = None
    model_type: Optional[str] = None
    distribution: Optional[str] = None
    link_function: Optional[str] = None
    verbosity: str = 'INFO'

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.analysis_types:
            raise ValueError("No analysis types specified")
            
        if AnalysisType.PREPARE_DATA in self.analysis_types:
            if not self.input_dir:
                raise ValueError("Input directory required for data preparation")
            if not self.questionnaire_file:
                raise ValueError("Questionnaire file required for data preparation")
            if not self.questionnaire_name:
                raise ValueError("Questionnaire name required for data preparation")
        else:
            if not self.input_file:
                raise ValueError("Input file required for analysis functions")

class ModelAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._setup_logging()
        self._setup_directories()
        self.data = None
        self.prepared_data_path = None
        
    def _setup_logging(self):
        """Configure logging based on verbosity level"""
        logging.basicConfig(
            level=getattr(logging, self.config.verbosity),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.logger.info(f"Output directory set to: {self.config.output_dir}")

    def run_exploratory_analysis(self):
        """Run exploratory data analysis"""
        self.logger.info("Running exploratory data analysis")
    
        try:
            input_file = self._get_input_path()
            run_eda(input_file, self.config.output_dir)
            self.logger.info(f"Exploratory data analysis complete. Results saved in {self.config.output_dir}")
        except Exception as e:
            self.logger.error(f"Error in exploratory data analysis: {str(e)}")
            raise
    
        return self

    def prepare_data(self):
        """Prepare data from directory of CSV files"""
        
        self.logger.info(f"Preparing data from directory: {self.config.input_dir}")
        
        try:
            # Check if input directory exists and contains CSV files
            if not os.path.isdir(self.config.input_dir):
                raise ValueError(f"Input directory does not exist: {self.config.input_dir}")
                
            csv_files = [f for f in os.listdir(self.config.input_dir) 
                        if f.endswith('.csv') and self.config.questionnaire_name.upper() in f.upper()]
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {self.config.input_dir} "
                               f"matching questionnaire name {self.config.questionnaire_name}")
            
            # Prepare the data
            self.data = prep_data(
                directory=self.config.input_dir,
                questionnaire_name=self.config.questionnaire_name
            )
            
            # Save prepared data
            self.prepared_data_path = os.path.join(
                self.config.output_dir,
                f"{self.config.questionnaire_name}_data.csv"
            )
            self.data.to_csv(self.prepared_data_path, index=False)
            
            self.logger.info(f"Data preparation complete. Saved to: {self.prepared_data_path}")
            
        except Exception as e:
            self.logger.error(f"Error during data preparation: {str(e)}")
            raise
        
        return self

    def plot_loglog_mean(self):
        """Generate log-log mean plots"""
        self.logger.info("Generating log-log mean plots")
        
        try:
            if self.config.questionnaire_file:
                human_baselines = load_mean_human_baselines(self.config.questionnaire_file)
            else:
                human_baselines = {}
                
            plot_data = prepare_mean_data(self._get_data())
            plot_scaling_behavior(plot_data, self.config.output_dir, human_baselines)
            
            self.logger.info("Log-log mean plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating log-log mean plots: {str(e)}")
            raise
            
        return self

    def plot_loglog_variance(self):
        """Generate log-log variance plots"""
        self.logger.info("Generating log-log variance plots")
        
        try:
            if self.config.questionnaire_file:
                human_baselines = load_var_human_baselines(self.config.questionnaire_file)
            else:
                human_baselines = {}
                
            variance_df = prepare_var_data(self._get_data())
            plot_variance_scaling(variance_df, self.config.output_dir, human_baselines)
            
            self.logger.info("Log-log variance plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating log-log variance plots: {str(e)}")
            raise
            
        return self

    def plot_radar(self):
        """Generate radar plots"""
        self.logger.info("Generating radar plots")
        
        try:
            create_faceted_radar_plot(
                self._get_data(),
                'score',
                'Model Performance Analysis by Persona',
                os.path.join(self.config.output_dir, 'radar_plot')
            )
            
            self.logger.info("Radar plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating radar plots: {str(e)}")
            raise
            
        return self

   
    def plot_violin(self):
        """Generate violin plots"""
        self.logger.info("Generating violin plots")

        try:
            plot_violin_data(
                self._get_data(),
                self.config.questionnaire_name,
                self.config.output_dir
            )
            
            self.logger.info("Violin plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating violin plots: {str(e)}")
            raise
            
        return self

    def run_regression(self):
        """Run regression analysis"""
        if not all([self.config.formula, self.config.model_type]):
            raise ValueError("Formula and model type are required for regression analysis")
            
        self.logger.info("Running regression analysis")
        
        try:
            output_file = os.path.join(self.config.output_dir, 'regression_results.txt')
            model, res = run_regression_analysis(
                input_file=self._get_input_path(),
                formula=self.config.formula,
                model_type=self.config.model_type,
                distr=self.config.distribution,
                link=self.config.link_function,
                output_file=output_file
            )
            
            self.logger.info("Regression analysis complete")
            return model, res
            
        except Exception as e:
            self.logger.error(f"Error in regression analysis: {str(e)}")
            raise

        return self

    def run_repeated_measures(self):
        """Run repeated measures analysis"""
        self.logger.info("Running repeated measures analysis")
        
        try:
            rm_main(self._get_input_path())
            self.logger.info("Repeated measures analysis complete")
        except Exception as e:
            self.logger.error(f"Error in repeated measures analysis: {str(e)}")
            raise
            
        return self

    def run_anova(self):
        """Run ANOVA analysis"""
        self.logger.info("Running ANOVA analysis")
        
        try:
            anova_main(self._get_input_path())
            self.logger.info("ANOVA analysis complete")
        except Exception as e:
            self.logger.error(f"Error in ANOVA analysis: {str(e)}")
            raise
            
        return self

    def run_all_analyses(self):
        """Run all specified analyses in sequence"""
        analysis_map = {
            AnalysisType.PREPARE_DATA: self.prepare_data,
            AnalysisType.EXPLORATORY: self.run_exploratory_analysis,
            AnalysisType.LOGLOG_MEAN: self.plot_loglog_mean,
            AnalysisType.LOGLOG_VARIANCE: self.plot_loglog_variance,
            AnalysisType.VIOLIN: self.plot_violin, 
            AnalysisType.RADAR: self.plot_radar,
            AnalysisType.REGRESSION: self.run_regression,
            AnalysisType.REPEATED_MEASURES: self.run_repeated_measures,
            AnalysisType.ANOVA: self.run_anova
        }

        for analysis_type in self.config.analysis_types:
            try:
                self.logger.info(f"Running {analysis_type.name} analysis...")
                analysis_map[analysis_type]()
                self.logger.info(f"Completed {analysis_type.name} analysis")
            except Exception as e:
                self.logger.error(f"Error in {analysis_type.name} analysis: {str(e)}")
                raise

    def _get_input_path(self) -> str:
        """Get the appropriate input path based on the analysis context"""
        if self.prepared_data_path:
            return self.prepared_data_path
        return self.config.input_file

    def _get_data(self) -> pd.DataFrame:
        """Get the data, either from prepared data or by loading from file"""
        if self.data is None:
            self.data = pd.read_csv(self._get_input_path())
        return self.data

# Example usage
if __name__ == "__main__":
    # Example configuration for data preparation and analysis
    config = AnalysisConfig(
        input_dir="raw_data",
        output_dir="results",
        questionnaire_file="questionnaire.json",
        questionnaire_name="BFI",
        analysis_types=[
            AnalysisType.PREPARE_DATA,
            AnalysisType.LOGLOG_MEAN,
            AnalysisType.RADAR
        ],
        verbosity="INFO"
    )

    # Create analyzer and run analyses
    analyzer = ModelAnalyzer(config)
    analyzer.run_all_analyses()
