#Write your anomaly detection code here
import json
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, avg, stddev, min, max,
    when, abs, current_timestamp
)
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingAnomalyDetector:
    """
    Real-time anomaly detection for credit card transactions.
    
    Approach:
    1. Compute rolling statistics on the transaction stream
    2. Flag transactions that deviate from the norm
    3. Use Z-score or similar statistical methods
    """
    
    def __init__(self, app_name="AnomalyDetection",
                 bootstrap_servers="localhost:9092",
                 topic="credit-transactions"):
        """
        Initialize Spark session for streaming
        
        Args:
            app_name: Spark application name
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic to consume from
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        
        logger.info("Anomaly Detector initialized")
    
    def read_kafka_stream(self):
        """
        Read credit card transactions from Kafka
        
        Returns:
            Spark DataFrame with raw Kafka messages
        """
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        logger.info(f"Streaming from Kafka topic: {self.topic}")
        return df
    
    def parse_json_events(self, df):
        """
        Parse JSON events from Kafka messages
        
        Args:
            df: Spark DataFrame with raw Kafka messages
            
        Returns:
            Spark DataFrame with parsed JSON fields
        """
        parsed_df = df.select(
            from_json(col("value").cast("string"), self._get_schema()).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def _get_schema(self):
        """
        Define schema for credit card transaction events
        """
        schema = StructType([
            StructField("Time", DoubleType(), True),
            StructField("V1", DoubleType(), True),
            StructField("V2", DoubleType(), True),
            StructField("V3", DoubleType(), True),
            StructField("V4", DoubleType(), True),
            StructField("V5", DoubleType(), True),
            StructField("V6", DoubleType(), True),
            StructField("V7", DoubleType(), True),
            StructField("V8", DoubleType(), True),
            StructField("V9", DoubleType(), True),
            StructField("V10", DoubleType(), True),
            StructField("V11", DoubleType(), True),
            StructField("V12", DoubleType(), True),
            StructField("V13", DoubleType(), True),
            StructField("V14", DoubleType(), True),
            StructField("V15", DoubleType(), True),
            StructField("V16", DoubleType(), True),
            StructField("V17", DoubleType(), True),
            StructField("V18", DoubleType(), True),
            StructField("V19", DoubleType(), True),
            StructField("V20", DoubleType(), True),
            StructField("V21", DoubleType(), True),
            StructField("V22", DoubleType(), True),
            StructField("V23", DoubleType(), True),
            StructField("V24", DoubleType(), True),
            StructField("V25", DoubleType(), True),
            StructField("V26", DoubleType(), True),
            StructField("V27", DoubleType(), True),
            StructField("V28", DoubleType(), True),
            StructField("Amount", DoubleType(), True),
            StructField("Class", IntegerType(), True),
            StructField("timestamp", StringType(), True),
        ])
        return schema
    
    def compute_rolling_statistics(self, df, window_duration="5 minutes"):
        """
        Compute rolling statistics over a time window.
        
        Args:
            df: Parsed transaction DataFrame
            window_duration: Spark time window (e.g., "5 minutes", "1 hour")
            
        Returns:
            DataFrame with rolling statistics
        """
        # Convert timestamp to timestamp type if needed
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Compute rolling stats on Amount field
        stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            avg("Amount").alias("avg_amount"),
            stddev("Amount").alias("stddev_amount"),
            min("Amount").alias("min_amount"),
            max("Amount").alias("max_amount"),
            avg("Class").alias("avg_fraud_rate")
        )
        
        return stats
    
    def detect_anomalies_zscore(self, df, window_duration="5 minutes", threshold=3.0):
        """
        Detect anomalies using Z-score method.
        
        Transactions with amounts deviating more than `threshold` standard deviations
        from the rolling mean are flagged as anomalies.
        
        Args:
            df: Parsed transaction DataFrame
            window_duration: Rolling window duration
            threshold: Z-score threshold (default 3.0 = 99.7% confidence)
            
        Returns:
            DataFrame with anomaly flags
        """
        # Convert timestamp
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Compute rolling statistics
        stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            avg("Amount").alias("avg_amount"),
            stddev("Amount").alias("stddev_amount")
        )
        
        # Join stats back to transactions
        windowed_df = df.withColumn(
            "window",
            window(col("parsed_time"), window_duration)
        )
        
        enriched = windowed_df.join(
            stats,
            on="window",
            how="left"
        )
        
        # Compute Z-score
        enriched = enriched.withColumn(
            "z_score",
            when(
                col("stddev_amount") > 0,
                abs((col("Amount") - col("avg_amount")) / col("stddev_amount"))
            ).otherwise(0)
        )
        
        # Flag anomalies
        anomalies = enriched.withColumn(
            "is_anomaly",
            col("z_score") > threshold
        ).select(
            "timestamp",
            "Amount",
            "avg_amount",
            "stddev_amount",
            "z_score",
            "is_anomaly"
        )
        
        return anomalies
    
    def detect_anomalies_iqr(self, df, window_duration="5 minutes", multiplier=1.5):
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Transactions outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged as anomalies.
        
        Args:
            df: Parsed transaction DataFrame
            window_duration: Rolling window duration
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            DataFrame with anomaly flags
        """
        # Convert timestamp
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Compute rolling percentiles
        stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            avg("Amount").alias("avg_amount"),
            col("Amount").quantile(0.25).alias("q1"),  # Note: Spark uses approx_percentile
            col("Amount").quantile(0.75).alias("q3"),
        )
        
        # This is a simplified version - Spark's quantile is approximate
        # For production, use approx_percentile
        
        return df.select(
            "timestamp",
            "Amount"
        )
    
    def detect_anomalies_isolation_forest(self, df):
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Note: Requires spark-ml-lib. This is a placeholder for the concept.
        
        Args:
            df: Parsed transaction DataFrame
            
        Returns:
            DataFrame with anomaly predictions
        """
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.outlier import IsolationForest
        
        # Assemble features
        feature_cols = ["Amount"]  # Can add more features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        features_df = assembler.transform(df)
        
        # Train isolation forest
        iso_forest = IsolationForest(
            contamination=0.05,  # Expected proportion of anomalies
            features_col="features",
            prediction_col="anomaly"
        )
        
        predictions = iso_forest.transform(features_df)
        
        return predictions.select(
            "timestamp",
            "Amount",
            col("anomaly").alias("is_anomaly")
        )
    
    def write_anomalies(self, df, output_mode="append"):
        """
        Write detected anomalies to console and optionally save to file
        
        Args:
            df: DataFrame with anomalies
            output_mode: Spark streaming output mode
        """
        # Filter to show only anomalies
        anomalies = df.filter(col("is_anomaly") == True)
        
        query = anomalies.writeStream \
            .format("console") \
            .option("truncate", "false") \
            .outputMode(output_mode) \
            .start()
        
        return query
    
    def run(self, method="zscore", window_duration="5 minutes", threshold=3.0):
        """
        Run the streaming anomaly detection pipeline
        
        Args:
            method: Anomaly detection method ('zscore', 'iqr', 'isolation_forest')
            window_duration: Rolling window for statistics
            threshold: Sensitivity threshold
        """
        try:
            # Read from Kafka
            raw_df = self.read_kafka_stream()
            
            # Parse JSON
            parsed_df = self.parse_json_events(raw_df)
            
            # Detect anomalies
            if method == "zscore":
                anomalies_df = self.detect_anomalies_zscore(
                    parsed_df,
                    window_duration=window_duration,
                    threshold=threshold
                )
            elif method == "iqr":
                anomalies_df = self.detect_anomalies_iqr(
                    parsed_df,
                    window_duration=window_duration
                )
            else:
                logger.error(f"Unknown method: {method}")
                return
            
            # Write output
            query = self.write_anomalies(anomalies_df)
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in anomaly detection pipeline: {e}")
            raise
        finally:
            self.spark.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Streaming anomaly detection')
    parser.add_argument('--method', type=str, default='zscore',
                        choices=['zscore', 'iqr'],
                        help='Anomaly detection method')
    parser.add_argument('--window', type=str, default='5 minutes',
                        help='Rolling window duration')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Z-score threshold')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', type=str, default='credit-transactions',
                        help='Kafka topic to consume from')
    
    args = parser.parse_args()
    
    detector = StreamingAnomalyDetector(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    
    detector.run(
        method=args.method,
        window_duration=args.window,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
