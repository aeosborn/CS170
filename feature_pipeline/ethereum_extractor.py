import os
import time
import pandas as pd
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import heapq
from requests.exceptions import HTTPError

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

class EthereumFeatureExtractor:
    def __init__(self, provider_list, n_observations=10000, data_directory="data"):
        """
        Initialize the Ethereum extractor with a provider URL.
        
        Args:
            provider_url (str): Ethereum node provider URL (e.g., Infura, Alchemy)
        """
        self.provider_carosel = iter([Web3(Web3.HTTPProvider(url)) for url in provider_list])
        
        self.w3 = next(self.provider_carosel)
        # REF : https://web3py.readthedocs.io/en/stable/middleware.html
        
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        # [ DEPRECATED ]
        # self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        self.n_observations = n_observations
        self.data_directory = data_directory
        # Add middleware for POA chains
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node using provider: {self.w3.provider.endpoint_uri}")
        logger.info(f"Connected to Ethereum node: {self.w3.client_version}")

    def provider_cycle(self):
        logger.info("Cycling provider")
        self.w3 = next(self.provider_carosel)
    
    def get_block_by_timestamp(self, target_timestamp):
        """
        Find the block closest to the given timestamp using binary search on block timestamps.
        
        Args:
            target_timestamp (int): Unix timestamp to search for
            
        Returns:
            int: Eth block number closest to the target timestamp
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
    
    def extract_transactions(self, start_time, end_time, observations=None) -> str:
        """
        Extract transactions between start_time and end_time.
        If observations is provided, only the top N transactions will be saved.
        
        Args:
            start_time (datetime or int): Start time (datetime or unix timestamp)
            end_time (datetime or int): End time (datetime or unix timestamp)
            observations (int, optional): Number of top transactions to keep
                
        Returns:
            str: Path to the saved transactions file
        """
        
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        
        # Get start/end blocks
        start_block = self.get_block_by_timestamp(start_time)
        end_block = self.get_block_by_timestamp(end_time)
        
        logger.info(f"Extracting transactions from block {start_block} to {end_block} ({end_block - start_block + 1} blocks)")
        
        all_transactions = []
        # If observations is set, use a min-heap to keep track of top transactions by value
        min_heap = []
        keep_top_n = observations is not None and observations > 0

        # Contains processed hashes
        transaction_hashes = set()
        
        filename = f"eth_transactions_{start_block}_{end_block}"
        if observations:
            filename += f"_top{observations}"
        filename += ".csv"
        file_path = os.path.join(self.data_directory, filename)
        
        blocks_to_process = list(range(start_block, end_block + 1))
        # Skip blocks we've already processed
        if all_transactions:
            processed_blocks = set(pd.DataFrame(all_transactions)['blockNumber'].unique())
            blocks_to_process = [b for b in blocks_to_process if b not in processed_blocks]
            logger.info(f"Skipping {len(processed_blocks)} already processed blocks")
        
        ## Iterate through blocks in the given start, end range
        try:
            for i, block_number in enumerate(tqdm(blocks_to_process, desc="Processing blocks")):
                try:
                    # Get Eth block for given block number
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    block_timestamp = datetime.fromtimestamp(block.timestamp)
                    
                    # Operate on each transaction
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
                            # tx_dict['logs'] = [log.args for log in receipt.logs] if hasattr(receipt, 'logs') else []
                            
                        except HTTPError as e:
                            if e.response.status_code == 429:
                                logger.warning(f"Rate limit exceeded while getting receipt for tx {tx_hash}: {e}")
                                logger.info("Cycling providers")
                                self.provider_cycle()
                            else:
                                logger.warning(f"HTTP error while getting receipt for tx {tx_hash}: {e}")
                        
                        except Exception as e:
                            logger.warning(f"Error getting receipt for tx {tx_hash}: {e}")
                            tx_dict['gasUsed'] = None
                            tx_dict['status'] = None
                            tx_dict['logs'] = []
                        
                        tx_dict['valueETH'] = self.w3.from_wei(tx_dict['value'], 'ether')
                        
                        if keep_top_n:
                            # Use the transaction value as the sorting key
                            value = float(tx_dict['valueETH'])
                            
                            if len(min_heap) < observations:
                                # Heap not full yet, add transaction 
                                heapq.heappush(min_heap, (value, tx_hash, tx_dict))
                            elif value > min_heap[0][0]:
                                heapq.heappushpop(min_heap, (value, tx_hash, tx_dict))
                        # else:
                        #     # If not using top observations, just add all transactions
                        #     all_transactions.append(tx_dict)
                        
                        transaction_hashes.add(tx_hash)
                
                except Exception as e:
                    logger.error(f"Error processing block {block_number}: {e}")
                    continue
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            # If using min-heap, extract all transactions into all_transactions list
            if keep_top_n and min_heap:
                # Sort by value in descending order (highest value first)
                min_heap.sort(reverse=True)
                all_transactions = [tx_data for _, _, tx_data in min_heap]
                logger.info(f"Kept top {len(all_transactions)} transactions by value")
            
            # Final save
            if all_transactions:
                self._save_transactions(all_transactions, file_path)
                logger.info(f"Saved {len(all_transactions)} transactions to {file_path}")
            else:
                logger.warning("No transactions extracted")
            
            return file_path
        
        except KeyboardInterrupt:
            logger.info("Extraction interrupted by user")
            
            # If using min-heap, extract transactions before saving
            if keep_top_n and min_heap:
                # Sort by value in descending order (highest value first)
                min_heap.sort(reverse=True)
                all_transactions = [tx_data for _, _, tx_data in min_heap]
            
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