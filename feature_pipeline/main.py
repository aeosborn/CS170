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


def generate_time_intervals(start_date, end_date, n_intervals):
    """
    Generate time intervals based on the specified parameters.
    
    Args:
        start_date (datetime): Global start date
        end_date (datetime): Global end date
        n_observations (int): Number of observations to divide the date range into
    
    Returns:
        list: List of (start_time, end_time) tuples
    """
    span = timedelta(days=(end_date - start_date).days)

    
    if span == timedelta(0):
        logger.warning("Start date and end date are the same. No intervals generated.")
        return [(start_date, end_date)]
    logger.info(f"Generating {n_intervals} intervals of {span.days} days.")

    delta = timedelta(days=span.days / n_intervals)

    intervals = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        intervals.append((current_start, current_end))
        current_start = current_end
    return intervals


# def load_intervals_from_file(file_path):
#     """
#     Load time intervals from a JSON file.
    
#     Args:
#         file_path (str): Path to the JSON file containing intervals
    
#     Returns:
#         list: List of (start_time, end_time) tuples
#     """
#     try:
#         with open(file_path, 'r') as f:
#             intervals_data = json.load(f)
        
#         intervals = []
#         for interval in intervals_data:
#             start = pd.to_datetime(interval['start'])
#             end = pd.to_datetime(interval['end'])
#             intervals.append((start, end))
        
#         return intervals
#     except Exception as e:
#         logger.error(f"Error loading intervals from {file_path}: {e}")
#         sys.exit(1)


# def save_intervals_to_file(intervals, file_path):
#     """
#     Save time intervals to a JSON file.
    
#     Args:
#         intervals (list): List of (start_time, end_time) tuples
#         file_path (str): Path to save the JSON file
#     """
#     intervals_data = []
#     for start, end in intervals:
#         intervals_data.append({
#             'start': start.isoformat(),
#             'end': end.isoformat()
#         })
    
#     with open(file_path, 'w') as f:
#         json.dump(intervals_data, f, indent=2)


def main():
    """Main entry point."""
    os.getenv('../.env')

    provider = os.getenv('ETHEREUM_PROVIDER_URL')
    start_date = datetime.strptime(os.getenv('START_DATE'), '%m/%d/%Y')
    end_date = datetime.strptime(os.getenv('END_DATE'), '%m/%d/%Y')
    observations = int(os.getenv('OBSERVATIONS'))
    operation_max_size = int(os.getenv('MAX_EXTRACTON_SIZE'))
    whale_transaction_minimum = int(os.getenv('WHALE_TRANSACTION_MINIMUM'))

    logger.info(f"Enviroment variables loaded.")
    
    if operation_max_size < 0:
        operation_max_size = observations

    
    # Create save directory for this run, and debugging files
    data_dir = os.path.join(os.getcwd(), f"ethereum_data{start_date}_{end_date}")
    os.makedirs(data_dir, exist_ok=True)
    results_file_string = os.path.join(data_dir, "results.csv")
    process_status_file = os.path.join(data_dir, "process_status.json")

    DEBUG = ( os.getenv('DEBUG').lower() == 'true' or os.getenv('DEBUG', 'false').lower() == '1' )
    
    intervals = generate_time_intervals(
        start_date=start_date, 
        end_date=end_date, 
        n_intervals=observations/operation_max_size,
        )
    
    # Save generated intervals for reference
    # intervals_file = os.path.join(data_dir, "time_intervals.json")
    # save_intervals_to_file(intervals, intervals_file)
    
    if DEBUG:
        logger.debug(f"Generated {len(intervals)} time intervals from {start_date} to {end_date}")
    # logger.info(f"Intervals saved to {intervals_file}")
    
    # Initialize the feature extractor
    extractor = EthereumFeatureExtractor(provider, n_observations=observations)
        
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
                end_time=interval_end
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
            
            # # Print summary if successful
            # if result["status"] == "success" and "summary" in result:
            #     summary = result["summary"]
            #     print(f"\nInterval {process_id+1} Summary:")
            #     print(f"Transactions: {summary['total_transactions']}")
            #     print(f"Blocks: {summary['blocks_processed']}")
            #     print(f"ETH Volume: {summary['total_eth_volume']:.2f}" if summary['total_eth_volume'] else "ETH Volume: N/A")
            #     print(f"Whale Wallets: {result['whale_wallets_count']}")
            #     print(f"Miner Wallets: {result['miner_wallets_count']}")
        
    
    # Generate merged results if we have more than one interval
    # if len(results) > 1:
    #     logger.info("Generating merged results")
    #     merged_results = {
    #         "total_intervals": len(results),
    #         "successful_intervals": sum(1 for r in results if r["status"] == "success"),
    #         "total_transactions": sum(r["summary"]["total_transactions"] for r in results if r["status"] == "success"),
    #         "total_blocks_processed": sum(r["summary"]["blocks_processed"] for r in results if r["status"] == "success"),
    #         "total_eth_volume": sum(r["summary"]["total_eth_volume"] for r in results if r["status"] == "success"),
    #         "total_whales_identified": sum(r["whale_wallets_count"] for r in results if r["status"] == "success"),
    #         "total_miners_identified": sum(r["miner_wallets_count"] for r in results if r["status"] == "success"),
    #     }
        
    # Print final summary

    results = pd.read_csv(process_status_file)

    print("\nOverall Summary:")
    print(f"Total intervals processed: {len(results)}")
    logger.info("Controller process completed")


if __name__ == "__main__":
    main()