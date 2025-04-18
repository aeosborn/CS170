import os
import time
import pandas as pd
from datetime import datetime
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
    def __init__(self, provider_list, n_observations=10000, results_file="data", delay=0.05, validator_extraction=True):
        """
        Initialize the Ethereum extractor with a provider URL.
        
        Args:
            provider_url (str): Ethereum node provider W3 URL
        """
        # Validation wallet address
        ## !! DO NOT CHANGE !! ##
        self.deposit_signature = "0x22895118"
        self.validator_wallet_address = "0x00000000219ab540356cBB839Cbe05303d7705Fa" # ETH2.0 Validation Deposit Addr
        # self.validator_wallet_address = "0x41e5560054824eA6B0732E656E3Ad64E20e94E45" ## TEST ADDR DO NOT USE
        
        self.do_validator_extraction = validator_extraction

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
                
                # Check if within 6 mins of target timestamp
                if abs(mid_timestamp - target_timestamp) < 360:
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
    
    def extract_transactions(self, start_time, end_time, observations=None) -> None:
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

        # Initialize datastructure for validation trasaction capture
        validators_dataset = []
        
        # Pre-allocate the set for tx hashes
        transaction_hashes = set()
        
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
                        
                        if tx_dict['to'] == self.validator_wallet_address: #or 
                            # tx_dict['input'].startswith(self.deposit_signature)):

                            # Add to validator dataset
                            validators_dataset.append(tx_dict)
                            logger.info(f"Validator transaction found: {tx_hash}. Validator cound: {len(validators_dataset)}")

                        # Add metadata
                        tx_dict['blockTimestamp'] = block_timestamp
                        tx_dict['datetime'] = block_datetime
                        
                        # Calculate ETH value once
                        value_eth = float(self.w3.from_wei(tx_dict['value'], 'ether'))
                        tx_dict['valueETH'] = value_eth
                        
                        # Only process transaction receipt and add to heap if value is high enough
                        if value_eth > current_min_value or len(min_heap) < observations:
                            # Get transaction receipt for additional data
                            try:
                                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                                tx_dict['gasUsed'] = receipt.gasUsed
                                tx_dict['status'] = receipt.status
                            except HTTPError as e:
                                if e.response.status_code == 429:
                                    logger.warning(f"Rate limit exceeded while getting receipt for tx {tx_hash}: {e}")
                                    logger.info("Cycling providers")
                                    self.provider_cycle()
                                    try:
                                        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                                        tx_dict['gasUsed'] = receipt.gasUsed
                                        tx_dict['status'] = receipt.status
                                    except Exception as inner_e:
                                        logger.warning(f"Error getting receipt for tx {tx_hash} after cycling providers: {inner_e}")
                                        tx_dict['gasUsed'] = None
                                        tx_dict['status'] = None
                                else:
                                    logger.warning(f"HTTP error while getting receipt for tx {tx_hash}: {e}")
                                    tx_dict['gasUsed'] = None
                                    tx_dict['status'] = None
                            except Exception as e:
                                logger.warning(f"Error getting receipt for tx {tx_hash}: {e}")
                                tx_dict['gasUsed'] = None
                                tx_dict['status'] = None
                            
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
            
            logger.info(f"Processed {len(transaction_hashes)} transactions from {start_block} to {end_block}")
            
            ## Sort and save collected transactions
            min_heap.sort(reverse=True)
            all_transactions = [tx_data for _, _, tx_data in min_heap if tx_data]
            logger.info(f"Kept top {len(all_transactions)} transactions by value")

            if all_transactions:
                self.save_transactions(all_transactions, self.results_file.replace(".csv", "_transactions.csv"))
                logger.info(f"Saved {len(all_transactions)} transactions to {self.results_file.replace('.csv', '_transactions.csv')}")
            else:
                logger.warning("No transactions extracted")

            if len(validators_dataset) != 0:
                self.save_transactions(validators_dataset, self.results_file.replace(".csv", "_validator_transactions.csv"))
                logger.info(f"Saved {len(validators_dataset)} validator transactions to {self.results_file.replace('.csv', '_validator_transactions.csv')}")
            else:
                logger.info("No validator transactions collected")
            
        except KeyboardInterrupt:
            logger.info("Extraction interrupted by user")

        except Exception as e:
            logger.error(f"ERROR: {str(e).uppper()}")

    def save_transactions(self, transactions, filename):
        """
        Save transactions to a CSV file, with replacement.
        
        Args:
            transactions (list): List of transaction dictionaries to save.
            filename (str): Name of the file to save the transactions.
        """
        if os.path.exists(filename):
            logger.warning(f"File {filename} already exists. Removing.")
            os.remove(filename)
        
        try:
            df = pd.DataFrame(transactions)
            df.to_csv(filename, index=False,header=True)
        except Exception as e:
            logger.error(f"Error saving to {filename}: {e}")
            raise