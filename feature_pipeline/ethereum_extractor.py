import os
import time
import pandas as pd
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from tqdm import tqdm
import logging
import heapq
from requests.exceptions import HTTPError

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
    def __init__(self, provider_list, n_observations=10000, results_file="data", delay=0.05):
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
        self.results_file = results_file
        
        self.delay = delay
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
        
        ## Binary searching for closest block to start
        while left <= right:
            mid = (left + right) // 2
            try:
                mid_block = self.w3.eth.get_block(mid)
                mid_timestamp = mid_block.timestamp
                
                # Check if within 10 mins of target timestamp
                if abs(mid_timestamp - target_timestamp) < 600:
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
        # Convert timestamps once
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        
        # Get blocks for the time range
        start_block = self.get_block_by_timestamp(start_time)
        end_block = self.get_block_by_timestamp(end_time)
        
        logger.info(f"Extracting transactions from block {start_block} to {end_block} ({end_block - start_block + 1} blocks)")
        
        # Use a fixed-size array for the heap if keeping top N transactions
        # Initialize with dummy values that will be replaced
        min_heap = [(0, "dummy", {}) for _ in range(observations)]
        heapq.heapify(min_heap)
        current_min_value = 0
        
        # Pre-allocate the set for tx hashes
        transaction_hashes = set()
        
        filename = f"eth_transactions_{start_block}_{end_block}"
        if observations:
            filename += f"_top{observations}"
        filename += ".csv"
        file_path = os.path.join(self.results_file, filename)
        
        blocks_to_process = range(start_block, end_block + 1)
        
        try:
            for block_number in tqdm(blocks_to_process, desc="Processing blocks"):
                
                
                try:
                    # Get block with all transactions
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    block_timestamp = block.timestamp
                    block_datetime = datetime.fromtimestamp(block_timestamp).isoformat()
                    
                    # Process all transactions in the block
                    for tx in block.transactions:
                        tx_dict = dict(tx)
                        tx_hash = tx_dict['hash'].hex()
                        
                        # Skip if already processed
                        if tx_hash in transaction_hashes:
                            continue
                        
                        transaction_hashes.add(tx_hash)
                        
                        # Convert bytes to hex strings (only for keys we know might be bytes)
                        for key in ('hash', 'blockHash', 'from', 'to', 'input', 'r', 's', 'v'):
                            if key in tx_dict and isinstance(tx_dict[key], bytes):
                                tx_dict[key] = tx_dict[key].hex()
                        
                        # Add metadata
                        tx_dict['blockTimestamp'] = block_timestamp
                        tx_dict['datetime'] = block_datetime
                        
                        # Calculate ETH value once
                        value_eth = float(self.w3.from_wei(tx_dict['value'], 'ether'))
                        tx_dict['valueETH'] = value_eth
                        
                        # Only process transaction receipt and add to heap if value is high enough
                        if value_eth > current_min_value or len(min_heap) < observations:
                            # Get transaction receipt for additional data
                            self._process_transaction_receipt(tx_dict, tx_hash)
                            
                            # Update the heap
                            if len(min_heap) < observations:
                                heapq.heappush(min_heap, (value_eth, tx_hash, tx_dict))
                            else:
                                heapq.heappushpop(min_heap, (value_eth, tx_hash, tx_dict))
                                # Update current minimum value for fast comparison
                                current_min_value = min_heap[0][0]
                    
                except Exception as e:
                    logger.error(f"Error processing block {block_number}: {e}")
                    continue
                
                time.sleep(self.delay)
            

            ## Sort and save collected transactions
            min_heap.sort(reverse=True)
            all_transactions = [tx_data for _, _, tx_data in min_heap if tx_data]
            logger.info(f"Kept top {len(all_transactions)} transactions by value")

            if all_transactions:
                self.save_transactions(all_transactions, file_path)
                logger.info(f"Saved {len(all_transactions)} transactions to {file_path}")
            else:
                logger.warning("No transactions extracted")
            
            return file_path
        
        except KeyboardInterrupt:
            logger.info("Extraction interrupted by user")
            # self.save_transactions(all_transactions, file_path)
            # logger.info(f"Saved {len(all_transactions)} transactions to {file_path}")
            # return file_path
        
    def save_transactions(self, transactions):
        """Save transactions to a CSV file"""
        if not os.path.exists(self.results_file):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, 'a') as f:
            f.write(','.join(transactions[0].keys()) + '\n')
            for tx in transactions:
                f.write(','.join(str(v) for v in tx.values()) + '\n')
        logger.info(f"Transactions saved to {self.results_file}")

    def _process_transaction_receipt(self, tx_dict, tx_hash):
        """Helper method to process transaction receipt"""
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            tx_dict['gasUsed'] = receipt.gasUsed
            tx_dict['status'] = receipt.status
        except HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit exceeded while getting receipt for tx {tx_hash}: {e}")
                logger.info("Cycling providers")
                self.provider_cycle()
            else:
                logger.warning(f"HTTP error while getting receipt for tx {tx_hash}: {e}")
            tx_dict['gasUsed'] = None
            tx_dict['status'] = None
        except Exception as e:
            logger.warning(f"Error getting receipt for tx {tx_hash}: {e}")
            tx_dict['gasUsed'] = None
            tx_dict['status'] = None
    
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