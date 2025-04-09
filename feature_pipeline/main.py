#!/usr/bin/env python3
"""
Ethereum Blockchain Feature Extraction Controller
================================================

This script controls the execution of the Ethereum feature extraction pipeline
across multiple time intervals. It handles command-line arguments and 
orchestrates the extraction and analysis process.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from ethereum_extractor import EthereumFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eth_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_time_intervals(start_date, end_date, interval_span_type='day', interval_span_length="1"):
    """
    Generate time intervals based on the specified parameters.
    
    Args:
        start_date (datetime): Global start date
        end_date (datetime): Global end date
        interval_span_type (str): Type of interval span ('day', 'week', 'month', 'year')
        interval_span_length (str or int): Length of each interval span
    
    Returns:
        list: List of (start_time, end_time) tuples
    """
    from datetime import timedelta
    import logging
    
    logger = logging.getLogger(__name__)
    
    if start_date > end_date:
        logger.error("Start date cannot be after end date.")
        return []
        
    if start_date == end_date:
        logger.warning("Start date and end date are the same. Returning single interval.")
        return [(start_date, end_date)]
    
    try:
        interval_span_length = int(interval_span_length)
    except ValueError:
        logger.error(f"Invalid interval_span_length: {interval_span_length}. Using default value of 1.")
        interval_span_length = 1
    
    # Get appropriate timedelta based on interval_span_type
    if interval_span_type.lower() == 'day':
        delta = timedelta(days=interval_span_length)
        span_name = "day" if interval_span_length == 1 else "days"
    elif interval_span_type.lower() == 'week':
        delta = timedelta(weeks=interval_span_length)
        span_name = "week" if interval_span_length == 1 else "weeks"
    elif interval_span_type.lower() == 'month':
        delta = timedelta(days=30 * interval_span_length)
        span_name = "month" if interval_span_length == 1 else "months"
    elif interval_span_type.lower() == 'year':
        delta = timedelta(days=365 * interval_span_length)
        span_name = "year" if interval_span_length == 1 else "years"
    else:
        logger.error(f"Unknown interval_span_type: {interval_span_type}. Using 'day' as default.")
        delta = timedelta(days=interval_span_length)
        span_name = "day" if interval_span_length == 1 else "days"
    
    logger.info(f"Generating intervals of {interval_span_length} {span_name}.")
    
    intervals = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        intervals.append((current_start, current_end))
        current_start = current_end
    
    return intervals

def main():
    """Main entry point."""
    os.getenv('../.env')

    provider_list = [str(os.getenv('ETHEREUM_PROVIDER_URL_LIST'))]
    start_date = datetime.strptime(os.getenv('START_DATE'), '%Y-%m-%d-%H:%M')
    end_date = datetime.strptime(os.getenv('END_DATE'), '%Y-%m-%d-%H:%M')
    observations_per_interval = int(os.getenv('OBSERVATIONS_PER_INVERVAL'))
    # operation_max_size = int(os.getenv('MAX_EXTRACTON_SIZE')) ## TODO
    ## Interval size/ span
    interval_span_type = 'day'
    interval_span_length="1"

    logger.info(f"Start Date: {start_date}")
    logger.info(f"End Date: {end_date}")

    logger.info(f"Enviroment variables loaded.")
    
    # Create save directory for this run, and debugging files
    data_dir = os.path.join(os.getcwd(), f"ethereum_data{start_date}_{end_date}")
    os.makedirs(data_dir, exist_ok=True)
    results_file_string = os.path.join(data_dir, "results.csv")
    process_status_file = os.path.join(data_dir, "process_status.json")

    DEBUG = ( os.getenv('DEBUG').lower() == 'true' or os.getenv('DEBUG', 'false').lower() == '1' )
    
    intervals = generate_time_intervals(
        start_date=start_date, 
        end_date=end_date, 
        interval_span_type=interval_span_type, 
        interval_span_length=interval_span_length,
    )
    
    # Initialize the feature extractor
    extractor = EthereumFeatureExtractor(provider_list=provider_list)
        
    # Process each time interval
    enum_intervals = list(enumerate(intervals))

    ## Initialized here to handle multithreading in the future, if this is used a true controller plane
    ## Loops will continue until the enum_intervals is empty
    ## enum_intervals is a queue of intervals to be processed
    ## Each interval should be small enough time-frame to be locally processed, then dumped

    while enum_intervals:
        process_id, (interval_start, interval_end) = enum_intervals.pop(0)
        logger.info(f"Processing interval {process_id+1}/{len(intervals)+1}: {interval_start} to {interval_end}")

        try:
            result = extractor.extract_transactions(
                start_time=interval_start,
                end_time=interval_end,
                observations=observations_per_interval
            )
            
            status = "success"

            with open(results_file_string, "a") as results_file:
                results_file.write(json.dumps(result) + "\n")

        except Exception as e:
            logger.error(f"Error processing interval {process_id+1}: {e}", exc_info=True)
            status = "error"
            result = f"Error: {str(e)}"
            
        finally:
            # Save result to process_status file
            with open(os.path.join(data_dir, "process_status.json"), "a") as status_file:
                json.dump({
                    "interval_id": process_id,
                    "start": interval_start.isoformat(),
                    "end": interval_end.isoformat(),
                    "status": status,
                    "result": result
                }, status_file)
                status_file.write("\n")

            logger.info(f"Completed interval {process_id+1}/{len(intervals)+1}")

    results = pd.read_csv(process_status_file)

    print("\nOverall Summary:")
    print(f"Total intervals processed: {len(results)}")
    logger.info("Controller process completed")

if __name__ == "__main__":
    main()