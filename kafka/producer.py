#Write your kafka producer code here
import json
import time
import argparse
from datetime import datetime
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditCardProducer:
    """Kafka producer for credit card transaction events"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='credit-transactions'):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka bootstrap server address
            topic: Kafka topic to publish to
        """
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        logger.info(f"Producer initialized for topic: {topic}")
    
    def load_data(self, csv_path):
        """
        Load credit card fraud data from CSV
        
        Args:
            csv_path: Path to the credit card fraud CSV file
            
        Returns:
            pandas.DataFrame with the data
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {csv_path}")
            return None
    
    def send_event(self, event, callback=None):
        """
        Send a single event to Kafka
        
        Args:
            event: Dictionary containing the transaction data
            callback: Optional callback function for success/failure handling
        """
        try:
            future = self.producer.send(self.topic, value=event)
            
            if callback:
                future.add_callback(callback)
                future.add_errback(lambda x: logger.error(f"Failed to send event: {x}"))
            
            # Wait for send to complete (optional - for synchronous behavior)
            # record_metadata = future.get(timeout=10)
            
        except Exception as e:
            logger.error(f"Error sending event: {e}")
    
    def stream_data(self, df, delay=0.1, batch_size=1):
        """
        Stream data from DataFrame to Kafka with configurable delay
        
        Args:
            df: pandas DataFrame with transaction data
            delay: Delay between events in seconds (default: 0.1 = 10 events/sec)
            batch_size: Number of events to send as a batch
        """
        logger.info(f"Starting to stream {len(df)} events with {delay}s delay between events")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # Convert row to dictionary
            event = row.to_dict()
            
            # Add timestamp if not present
            if 'timestamp' not in event:
                event['timestamp'] = datetime.utcnow().isoformat()
            
            # Send event
            self.send_event(event)
            
            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"Sent {idx + 1} events...")
            
            # Add delay
            time.sleep(delay)
        
        logger.info(f"Finished streaming {len(df)} events")
        self.close()
    
    def close(self):
        """Close the Kafka producer"""
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed")


def main():
    """Main entry point for the producer"""
    parser = argparse.ArgumentParser(description='Kafka producer for credit card transactions')
    parser.add_argument('--csv', type=str, default='creditcard.csv',
                        help='Path to credit card fraud CSV file')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', type=str, default='credit-transactions',
                        help='Kafka topic to publish to')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between events in seconds')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of events to stream')
    
    args = parser.parse_args()
    
    # Initialize producer
    producer = CreditCardProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    
    # Load data
    df = producer.load_data(args.csv)
    if df is None:
        return
    
    # Limit if specified
    if args.limit:
        df = df.head(args.limit)
    
    # Stream data
    try:
        producer.stream_data(df, delay=args.delay)
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
        producer.close()


if __name__ == '__main__':
    main()
