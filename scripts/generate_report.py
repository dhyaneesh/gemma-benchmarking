import argparse
import logging
from pathlib import Path

from src.visualization.plotter import BenchmarkPlotter

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/visualization.log")
        ]
    )

def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(description="Generate visualization reports from benchmark results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to save visualization reports"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Name for the output report (without extension)"
    )
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create plotter
        plotter = BenchmarkPlotter(args.output_dir)
        
        # Generate summary report and visualizations
        logger.info(f"Generating report from results in {args.results_dir}")
        plotter.generate_summary_report(args.results_dir, args.output_name)
        logger.info(f"Report generated successfully in {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

if __name__ == "__main__":
    main() 