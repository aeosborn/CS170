import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eth_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EthereumFeatureExtractor:
    def __init__(self,  provider_url, n_observations=10000,):
        """
        Initialize the Ethereum extractor with a provider URL.
        
        Args:
            provider_url (str): Ethereum node provider URL (e.g., Infura, Alchemy)
        """
            
        self.n_observations = n_observations
        
        # self.provider_url = provider_url
        self.w3 = Web3(Web3.HTTPProvider(provider_url))

        # Add middleware for POA chains compatibility (if needed)
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        logger.info(f"Connected to Ethereum node: {self.w3.client_version}")

        self.data_dir = os.path.join(os.getcwd(), f"ethereum_data{self.start_date}_{self.end_date}")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def timestamp_to_datetime(self, timestamp):
        """Convert Ethereum block timestamp to datetime."""
        return datetime.fromtimestamp(timestamp)
    
    def get_block_by_timestamp(self, target_timestamp):
        """
        Find the block closest to the given timestamp using binary search.
        
        Args:
            target_timestamp (int): Unix timestamp to search for
            
        Returns:
            int: Block number closest to the target timestamp
        """
        latest_block = self.w3.eth.block_number
        left, right = 1, latest_block
        
        logger.info(f"Searching for block at timestamp {target_timestamp} (approx. {datetime.fromtimestamp(target_timestamp)})")
        
        while left <= right:
            mid = (left + right) // 2
            try:
                mid_block = self.w3.eth.get_block(mid)
                mid_timestamp = mid_block.timestamp
                
                # Check if we're close enough (within an hour)
                if abs(mid_timestamp - target_timestamp) < 3600:
                    logger.info(f"Found block {mid} with timestamp {mid_timestamp} ({datetime.fromtimestamp(mid_timestamp)})")
                    return mid
                
                if mid_timestamp < target_timestamp:
                    left = mid + 1
                else:
                    right = mid - 1
            except Exception as e:
                logger.error(f"Error getting block {mid}: {e}")
                # If error, try to narrow the search space
                right = (left + right) // 2
        
        # Return the closest block we found
        closest_block = left if left <= latest_block else latest_block
        block_data = self.w3.eth.get_block(closest_block)
        logger.info(f"Closest block found: {closest_block} with timestamp {block_data.timestamp} ({datetime.fromtimestamp(block_data.timestamp)})")
        return closest_block
    
    def extract_transactions(self, save_interval=100):
        """
        Extract transactions between start_time and end_time.
        
        Args:
            start_time (datetime or int): Start time (datetime or unix timestamp)
            end_time (datetime or int): End time (datetime or unix timestamp)
            max_blocks (int, optional): Maximum number of blocks to process
            save_interval (int): Number of blocks after which to save intermediate results
            
        Returns:
            str: Path to the saved transactions file
        """

        max_blocks = self.n_observations
        start_time = pd.to_datetime(self.start_date)
        end_time = pd.to_datetime(self.end_date)

        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        
        # Get start/end blocks
        start_block = self.get_block_by_timestamp(start_time)
        end_block = self.get_block_by_timestamp(end_time)
        
        if max_blocks and (end_block - start_block + 1) > max_blocks:
            logger.warning(f"Range too large ({end_block - start_block + 1} blocks). Limiting to {max_blocks} blocks from start.")
            end_block = start_block + max_blocks - 1
        
        logger.info(f"Extracting transactions from block {start_block} to {end_block} ({end_block - start_block + 1} blocks)")
        
        all_transactions = []
        transaction_hashes = set()  # To avoid duplicates
        
        filename = f"eth_transactions_{start_block}_{end_block}.csv"
        file_path = os.path.join(self.data_dir, filename)
        
        # Check if we have intermediate results to resume from
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                all_transactions = existing_df.to_dict('records')
                transaction_hashes = set(existing_df['hash'].tolist())
                logger.info(f"Loaded {len(all_transactions)} existing transactions from {file_path}")
        
        blocks_to_process = list(range(start_block, end_block + 1))
        # Skip blocks we've already processed
        if all_transactions:
            processed_blocks = set(pd.DataFrame(all_transactions)['blockNumber'].unique())
            blocks_to_process = [b for b in blocks_to_process if b not in processed_blocks]
            logger.info(f"Skipping {len(processed_blocks)} already processed blocks")
        
        try:
            for i, block_number in enumerate(tqdm(blocks_to_process, desc="Processing blocks")):
                try:
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    block_timestamp = self.timestamp_to_datetime(block.timestamp)
                    
                    for tx in block.transactions:
                        tx_dict = dict(tx)
                        tx_hash = tx_dict['hash'].hex()
                        
                        # Skip if we've already processed this transaction
                        if tx_hash in transaction_hashes:
                            continue
                        
                        # Convert bytes to hex strings
                        for key, value in tx_dict.items():
                            if isinstance(value, bytes):
                                tx_dict[key] = value.hex()
                        
                        # Add additional metadata
                        tx_dict['blockTimestamp'] = block.timestamp
                        tx_dict['datetime'] = block_timestamp.isoformat()
                        
                        # Get transaction receipt for additional data (gas used, status)
                        try:
                            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                            tx_dict['gasUsed'] = receipt.gasUsed
                            tx_dict['status'] = receipt.status
                            
                            # Extract logs (can be useful for token transfers)
                            tx_dict['logs'] = [log.args for log in receipt.logs] if hasattr(receipt, 'logs') else []
                            
                        except Exception as e:
                            logger.warning(f"Error getting receipt for tx {tx_hash}: {e}")
                            tx_dict['gasUsed'] = None
                            tx_dict['status'] = None
                            tx_dict['logs'] = []
                        
                        # Calculate transaction value in ETH
                        tx_dict['valueETH'] = self.w3.from_wei(tx_dict['value'], 'ether')
                        
                        all_transactions.append(tx_dict)
                        transaction_hashes.add(tx_hash)
                    
                    # Save intermediate results at intervals
                    if (i + 1) % save_interval == 0:
                        self._save_transactions(all_transactions, file_path)
                        logger.info(f"Saved {len(all_transactions)} transactions after processing {i+1}/{len(blocks_to_process)} blocks")
                
                except Exception as e:
                    logger.error(f"Error processing block {block_number}: {e}")
                    continue
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Final save
            if all_transactions:
                self._save_transactions(all_transactions, file_path)
                logger.info(f"Saved {len(all_transactions)} transactions to {file_path}")
            else:
                logger.warning("No transactions extracted")
            
            return file_path
        
        except KeyboardInterrupt:
            logger.info("Extraction interrupted by user")
            if all_transactions:
                self._save_transactions(all_transactions, file_path)
                logger.info(f"Saved {len(all_transactions)} transactions to {file_path}")
            return file_path
    
    def _save_transactions(self, transactions, file_path):
        """Save to CSV"""
        try:
            df = pd.DataFrame(transactions)
            # Convert any remaining complex objects to strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x)
            
            df.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"Error saving transactions: {e}")
            # Try a more robust save method
            try:
                simple_df = pd.DataFrame([{k: str(v) for k, v in tx.items()} for tx in transactions])
                simple_df.to_csv(file_path, index=False)
            except Exception as e2:
                logger.error(f"Failed backup save method: {e2}")
    
    def identify_whales(self, transactions_file, min_value_eth=100, min_transactions=5):
        """
        Identify whale wallets based on high-value transactions.
        
        Args:
            transactions_file (str): Path to transactions CSV file
            min_value_eth (float): Minimum transaction value in ETH to consider
            min_transactions (int): Minimum number of transactions to qualify as a whale
            
        Returns:
            pd.DataFrame: DataFrame of identified whale wallets and their metrics
        """
        logger.info(f"Identifying whale wallets from {transactions_file}")
        
        df = pd.read_csv(transactions_file)
        
        # Filter whale transactions
        high_value_txs = df[df['valueETH'] >= min_value_eth]
        
        # Get whale Senders
        from_addr_counts = high_value_txs['from'].value_counts()
        from_whales = from_addr_counts[from_addr_counts >= min_transactions].index.tolist()
        
        # Get whale Revivers
        to_addr_counts = high_value_txs['to'].value_counts()
        to_whales = to_addr_counts[to_addr_counts >= min_transactions].index.tolist()
        
        all_whales = list(set(from_whales + to_whales))
        
        # Calculate wallet metrics
        whale_metrics = []
        for wallet in all_whales:
            sent = df[df['from'] == wallet]
            received = df[df['to'] == wallet]
            
            metrics = {
                'wallet': wallet,
                'total_sent_eth': sent['valueETH'].sum(),
                'total_received_eth': received['valueETH'].sum(),
                'num_sent_tx': len(sent),
                'num_received_tx': len(received),
                'avg_sent_eth': sent['valueETH'].mean() if len(sent) > 0 else 0,
                'avg_received_eth': received['valueETH'].mean() if len(received) > 0 else 0,
                'max_sent_eth': sent['valueETH'].max() if len(sent) > 0 else 0,
                'max_received_eth': received['valueETH'].max() if len(received) > 0 else 0,
                'first_seen': df[(df['from'] == wallet) | (df['to'] == wallet)]['datetime'].min(),
                'last_seen': df[(df['from'] == wallet) | (df['to'] == wallet)]['datetime'].max(),
            }
            
            # Net flow (positive means receiving more than sending)
            metrics['net_flow_eth'] = metrics['total_received_eth'] - metrics['total_sent_eth']
            
            whale_metrics.append(metrics)
        
        whale_df = pd.DataFrame(whale_metrics)
        
        # Save results
        output_file = os.path.join(self.data_dir, f"whale_wallets.csv")
        whale_df.to_csv(output_file, index=False)
        logger.info(f"Identified {len(whale_df)} whale wallets, saved to {output_file}")
        
        return whale_df
    
    def identify_miners(self, transactions_file, miner_analysis_window=1000):
        """
        Identify likely miner wallets based on transaction patterns.
        
        Args:
            transactions_file (str): Path to transactions CSV file
            miner_analysis_window (int): Number of blocks to analyze for mining patterns
            
        Returns:
            pd.DataFrame: DataFrame of identified miner wallets and their metrics
        """
        logger.info(f"Identifying mining wallets from {transactions_file}")
        
        df = pd.read_csv(transactions_file)
        
        # Group transactions by block to find block rewards
        block_data = []
        
        # Get unique blocks
        unique_blocks = df['blockNumber'].unique()
        logger.info(f"Analyzing {len(unique_blocks)} blocks for mining patterns")
        
        # For each block, get the miner address
        for block_num in tqdm(unique_blocks[:miner_analysis_window], desc="Analyzing blocks for miners"):
            try:
                block = self.w3.eth.get_block(int(block_num))
                
                # In PoW blocks, the miner is in coinbase/miner field
                if hasattr(block, 'miner'):
                    miner_address = block.miner
                elif hasattr(block, 'coinbase'):
                    miner_address = block.coinbase
                else:
                    # For PoS blocks, the fee recipient might be the closest equivalent
                    miner_address = None
                
                if miner_address:
                    block_info = {
                        'blockNumber': block_num,
                        'miner': miner_address,
                        'timestamp': block.timestamp,
                        'datetime': datetime.fromtimestamp(block.timestamp).isoformat()
                    }
                    block_data.append(block_info)
            except Exception as e:
                logger.warning(f"Error getting miner for block {block_num}: {e}")
        
        if not block_data:
            logger.warning("No miner data could be extracted")
            return pd.DataFrame()
        
        miners_df = pd.DataFrame(block_data)
        
        # Count blocks mined by each address
        miner_counts = miners_df['miner'].value_counts().reset_index()
        miner_counts.columns = ['wallet', 'blocks_mined']
        
        # Calculate percentage of analyzed blocks
        miner_counts['percentage_of_analyzed_blocks'] = (miner_counts['blocks_mined'] / len(miners_df)) * 100
        
        # Get first and last seen timestamps
        miner_first_seen = miners_df.groupby('miner')['datetime'].min().reset_index()
        miner_first_seen.columns = ['wallet', 'first_seen']
        
        miner_last_seen = miners_df.groupby('miner')['datetime'].max().reset_index()
        miner_last_seen.columns = ['wallet', 'last_seen']
        
        # Merge all metrics
        miner_metrics = miner_counts.merge(miner_first_seen, on='wallet').merge(miner_last_seen, on='wallet')
        
        # Add transaction patterns for each miner
        all_miner_metrics = []
        for _, miner in miner_metrics.iterrows():
            wallet = miner['wallet']
            
            # Get transactions sent and received
            sent = df[df['from'] == wallet]
            received = df[df['to'] == wallet]
            
            additional_metrics = {
                'total_sent_eth': sent['valueETH'].sum() if 'valueETH' in sent.columns else 0,
                'total_received_eth': received['valueETH'].sum() if 'valueETH' in received.columns else 0,
                'num_sent_tx': len(sent),
                'num_received_tx': len(received),
                'avg_sent_eth': sent['valueETH'].mean() if 'valueETH' in sent.columns and len(sent) > 0 else 0,
                'avg_received_eth': received['valueETH'].mean() if 'valueETH' in received.columns and len(received) > 0 else 0,
            }
            
            # Combine with mining metrics
            combined_metrics = {**miner.to_dict(), **additional_metrics}
            all_miner_metrics.append(combined_metrics)
        
        final_miners_df = pd.DataFrame(all_miner_metrics)
        
        # Save results
        output_file = os.path.join(self.data_dir, "miner_wallets.csv")
        final_miners_df.to_csv(output_file, index=False)
        logger.info(f"Identified {len(final_miners_df)} mining wallets, saved to {output_file}")
        
        return final_miners_df
    
    def analyze_transactions(self, wallet_address, transactions_file):
        """
        Analyze transactions for a specific wallet.
        
        Args:
            wallet_address (str): Ethereum wallet address
            transactions_file (str): Path to transactions CSV file
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing transactions for wallet {wallet_address}")
        
        df = pd.read_csv(transactions_file)
        
        # Filter transactions involving this wallet
        sent = df[df['from'] == wallet_address]
        received = df[df['to'] == wallet_address]
        
        # Basic statistics
        metrics = {
            'wallet': wallet_address,
            'total_sent_eth': sent['valueETH'].sum() if 'valueETH' in sent.columns else 0,
            'total_received_eth': received['valueETH'].sum() if 'valueETH' in received.columns else 0,
            'num_sent_tx': len(sent),
            'num_received_tx': len(received),
            'net_flow_eth': (received['valueETH'].sum() if 'valueETH' in received.columns else 0) -
                           (sent['valueETH'].sum() if 'valueETH' in sent.columns else 0),
            'unique_sent_to': sent['to'].nunique() if 'to' in sent.columns else 0,
            'unique_received_from': received['from'].nunique() if 'from' in received.columns else 0,
        }
        
        if not sent.empty:
            metrics['first_sent'] = sent['datetime'].min()
            metrics['last_sent'] = sent['datetime'].max()
            metrics['avg_sent_eth'] = sent['valueETH'].mean() if 'valueETH' in sent.columns else 0
            metrics['max_sent_eth'] = sent['valueETH'].max() if 'valueETH' in sent.columns else 0
        
        if not received.empty:
            metrics['first_received'] = received['datetime'].min()
            metrics['last_received'] = received['datetime'].max()
            metrics['avg_received_eth'] = received['valueETH'].mean() if 'valueETH' in received.columns else 0
            metrics['max_received_eth'] = received['valueETH'].max() if 'valueETH' in received.columns else 0
        
        # Time-based analysis (group by day)
        if 'datetime' in df.columns:
            all_wallet_tx = df[(df['from'] == wallet_address) | (df['to'] == wallet_address)].copy()
            
            if not all_wallet_tx.empty:
                all_wallet_tx['date'] = pd.to_datetime(all_wallet_tx['datetime']).dt.date
                daily_activity = all_wallet_tx.groupby('date').size().reset_index(name='tx_count')
                metrics['daily_avg_tx'] = daily_activity['tx_count'].mean()
                metrics['max_daily_tx'] = daily_activity['tx_count'].max()
                metrics['active_days'] = len(daily_activity)
        
        # Identify top trading addrs
        if not sent.empty and 'to' in sent.columns:
            top_recipients = sent['to'].value_counts().head(5).to_dict()
            metrics['top_recipients'] = top_recipients
        
        if not received.empty and 'from' in received.columns:
            top_senders = received['from'].value_counts().head(5).to_dict()
            metrics['top_senders'] = top_senders
        
        return metrics
    
    # def run_full_pipeline(self, start_time, end_time, max_blocks=None, 
    #                      min_whale_value_eth=100, min_whale_tx=5, 
    #                      wallet_to_analyze=None):
    #     """
    #     Run the full extraction and analysis pipeline.
        
    #     Args:
    #         start_time (datetime or int): Start time (datetime or unix timestamp)
    #         end_time (datetime or int): End time (datetime or unix timestamp)
    #         max_blocks (int, optional): Maximum number of blocks to process
    #         min_whale_value_eth (float): Minimum transaction value to consider for whale detection
    #         min_whale_tx (int): Minimum number of transactions to qualify as a whale
    #         wallet_to_analyze (str, optional): Specific wallet to analyze in detail
            
    #     Returns:
    #         dict: Summary of all analysis results
    #     """
    #     logger.info(f"Starting full Ethereum extraction pipeline from {start_time} to {end_time}")
        
    #     # Extract transactions
    #     transactions_file = self.extract_transactions(start_time, end_time, max_blocks)
        
    #     # Skip analysis if no transactions were extracted
    #     if not os.path.exists(transactions_file) or os.path.getsize(transactions_file) == 0:
    #         logger.warning("No transactions extracted, skipping analysis")
    #         return {"status": "error", "message": "No transactions extracted"}
        
    #     # Identify whale wallets
    #     whale_df = self.identify_whales(transactions_file, min_whale_value_eth, min_whale_tx)
        
    #     # Identify miner wallets
    #     miner_df = self.identify_miners(transactions_file)
        
    #     results = {
    #         "status": "success",
    #         "transactions_file": transactions_file,
    #         "whale_wallets_count": len(whale_df),
    #         "miner_wallets_count": len(miner_df),
    #         "whale_wallets_file": os.path.join(self.data_dir, "whale_wallets.csv"),
    #         "miner_wallets_file": os.path.join(self.data_dir, "miner_wallets.csv"),
    #     }
        
    #     # Optional: analyze specific wallet
    #     if wallet_to_analyze:
    #         wallet_analysis = self.analyze_transactions(wallet_to_analyze, transactions_file)
    #         results["wallet_analysis"] = wallet_analysis
            
    #         # Save the wallet analysis
    #         wallet_file = os.path.join(self.data_dir, f"wallet_{wallet_to_analyze[:10]}_analysis.json")
    #         with open(wallet_file, 'w') as f:
    #             json.dump(wallet_analysis, f, indent=2, default=str)
    #         results["wallet_analysis_file"] = wallet_file
        
    #     # Generate summary report
    #     summary = {
    #         "extraction_period": {
    #             "start": str(start_time),
    #             "end": str(end_time)
    #         },
    #         "blocks_processed": None,  # We'll calculate this
    #         "total_transactions": None,  # We'll calculate this
    #         "total_eth_volume": None,  # We'll calculate this
    #         "top_whales": whale_df.sort_values('total_sent_eth', ascending=False).head(10)[['wallet', 'total_sent_eth', 'total_received_eth', 'net_flow_eth']].to_dict('records') if not whale_df.empty else [],
    #         "top_miners": miner_df.sort_values('blocks_mined', ascending=False).head(10)[['wallet', 'blocks_mined', 'percentage_of_analyzed_blocks']].to_dict('records') if not miner_df.empty else []
    #     }
        
    #     try:
    #         tx_df = pd.read_csv(transactions_file)
    #         summary["blocks_processed"] = tx_df['blockNumber'].nunique()
    #         summary["total_transactions"] = len(tx_df)
    #         summary["total_eth_volume"] = tx_df['valueETH'].sum() if 'valueETH' in tx_df.columns else None
            
    #         # Real time period
    #         if 'datetime' in tx_df.columns:
    #             summary["actual_time_period"] = {
    #                 "first_tx": tx_df['datetime'].min(),
    #                 "last_tx": tx_df['datetime'].max()
    #             }
    #     except Exception as e:
    #         logger.error(f"Error calculating summary metrics: {e}")
        
    #     summary_file = os.path.join(self.data_dir, "extraction_summary.json")
    #     with open(summary_file, 'w') as f:
    #         json.dump(summary, f, indent=2, default=str)
        
    #     results["summary_file"] = summary_file
    #     results["summary"] = summary
        
    #     logger.info(f"Pipeline completed successfully. Results in {self.data_dir}")
    #     return results


# def main():
#     parser = argparse.ArgumentParser(description='Ethereum Blockchain Feature Extraction')
#     parser.add_argument('--provider', type=str, help='Ethereum provider URL (or set ETHEREUM_PROVIDER_URL env var)')
#     parser.add_argument('--start-time', type=str, required=True, help='Start time (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
#     parser.add_argument('--end-time', type=str, required=True, help='End time (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
#     parser.add_argument('--max-blocks', type=int, help='Maximum number of blocks to process')
#     parser.add_argument('--min-whale-eth', type=float, default=100, help='Minimum ETH value for whale detection')
#     parser.add_argument('--min-whale-tx', type=int, default=5, help='Minimum transactions for whale detection')
#     parser.add_argument('--analyze-wallet', type=str, help='Specific wallet to analyze in detail')
    
#     args = parser.parse_args()
    
#     # Parse datetime strings
#     try:
#         start_time = pd.to_datetime(args.start_time)
#         end_time = pd.to_datetime(args.end_time)
#     except Exception as e:
#         logger.error(f"Error parsing date/time: {e}")
#         logger.info("Please use format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
#         return
    
#     try:
#         # Initialize extractor
#         extractor = (args.provider)
        
#         # Run pipeline
#         results = extractor.run_full_pipeline(
#             start_time, 
#             end_time, 
#             max_blocks=args.max_blocks,
#             min_whale_value_eth=args.min_whale_eth,
#             min_whale_tx=args.min_whale_tx,
#             wallet_to_analyze=args.analyze_wallet
#         )
        
#         if results["status"] == "success":
#             print(f"\nPipeline completed successfully. Results saved to {extractor.data_dir}")
#             print(f"Processed {results['summary']['total_transactions']} transactions across {results['summary']['blocks_processed']} blocks")
#             print(f"Identified {results['whale_wallets_count']} whale wallets and {results['miner_wallets_count']} miner wallets")
            
#             if args.analyze_wallet:
#                 print(f"\nWallet analysis for {args.analyze_wallet}:")
#                 analysis = results["wallet_analysis"]
#                 print(f"Total sent: {analysis['total_sent_eth']:.2f} ETH in {analysis['num_sent_tx']} transactions")
#                 print(f"Total received: {analysis['total_received_eth']:.2f} ETH in {analysis['num_received_tx']} transactions")
#                 print(f"Net flow: {analysis['net_flow_eth']:.2f} ETH")
#         else:
#             print(f"\nPipeline failed: {results['message']}")
    
#     except Exception as e:
#         logger.error(f"Pipeline error: {e}", exc_info=True)
#         print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()