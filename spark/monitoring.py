#Write your monitoring code here

import json
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, avg, count, min, max,
    stddev, current_timestamp, when, abs, percentile_approx
)
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitor concept drift and model performance degradation in real-time.
    
    Key Metrics:
    1. Data Drift: Distribution changes in input features
    2. Label Drift: Changes in fraud rate
    3. Prediction Drift: Changes in model predictions
    4. Performance Drift: Degradation of model accuracy
    """
    
    def __init__(self, app_name="DriftMonitoring",
                 bootstrap_servers="localhost:9092",
                 topic="credit-transactions"):
        """
        Initialize Spark session for monitoring
        
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
        self.model = None
        
        logger.info("Drift Monitor initialized")
    
    def load_model(self, model_path):
        """
        Load pre-trained model for monitoring predictions
        
        Args:
            model_path: Path to saved model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
    
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
    
    def monitor_data_drift(self, df, window_duration="10 minutes"):
        """
        Monitor changes in input data distribution.
        
        Tracks: mean, std, min, max of key features over time
        
        Args:
            df: Parsed transaction DataFrame
            window_duration: Time window for monitoring
            
        Returns:
            DataFrame with data statistics
        """
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Monitor Amount field
        data_stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            count("*").alias("transaction_count"),
            avg("Amount").alias("avg_amount"),
            stddev("Amount").alias("stddev_amount"),
            min("Amount").alias("min_amount"),
            max("Amount").alias("max_amount"),
            percentile_approx("Amount", 0.5).alias("median_amount"),
            percentile_approx("Amount", 0.95).alias("p95_amount"),
            percentile_approx("Amount", 0.99).alias("p99_amount")
        )
        
        return data_stats
    
    def monitor_label_drift(self, df, window_duration="10 minutes"):
        """
        Monitor changes in fraud rate (label distribution).
        
        Args:
            df: Parsed transaction DataFrame
            window_duration: Time window for monitoring
            
        Returns:
            DataFrame with fraud rate statistics
        """
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        label_stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            count("*").alias("total_transactions"),
            sum(col("Class")).alias("fraud_count"),
            (avg("Class") * 100).alias("fraud_rate_percent")
        )
        
        return label_stats
    
    def monitor_prediction_drift(self, df, window_duration="10 minutes"):
        """
        Monitor changes in model predictions over time.
        
        Tracks: average predicted fraud probability, variance, etc.
        
        Args:
            df: DataFrame with model predictions (requires 'fraud_probability' column)
            window_duration: Time window for monitoring
            
        Returns:
            DataFrame with prediction statistics
        """
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Ensure predictions exist
        if "fraud_probability" not in df.columns:
            logger.warning("No fraud_probability column found. Skipping prediction drift.")
            return None
        
        pred_stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            count("*").alias("predictions_count"),
            avg("fraud_probability").alias("avg_fraud_prob"),
            stddev("fraud_probability").alias("stddev_fraud_prob"),
            percentile_approx("fraud_probability", 0.5).alias("median_fraud_prob"),
            percentile_approx("fraud_probability", 0.95).alias("p95_fraud_prob")
        )
        
        return pred_stats
    
    def compute_performance_metrics(self, df, window_duration="10 minutes"):
        """
        Compute model performance metrics on streaming data.
        
        Uses actual labels (Class) vs predictions for accuracy, precision, recall.
        
        Args:
            df: DataFrame with predictions and actual labels
            window_duration: Time window for monitoring
            
        Returns:
            DataFrame with performance metrics
        """
        df = df.withColumn("parsed_time", col("timestamp").cast("timestamp"))
        
        # Ensure required columns exist
        if "is_fraud" not in df.columns or "Class" not in df.columns:
            logger.warning("Missing prediction or label columns")
            return None
        
        # True Positives, False Positives, False Negatives, True Negatives
        perf_stats = df.groupBy(
            window(col("parsed_time"), window_duration)
        ).agg(
            count(when((col("is_fraud") == 1) & (col("Class") == 1), True)).alias("tp"),
            count(when((col("is_fraud") == 1) & (col("Class") == 0), True)).alias("fp"),
            count(when((col("is_fraud") == 0) & (col("Class") == 1), True)).alias("fn"),
            count(when((col("is_fraud") == 0) & (col("Class") == 0), True)).alias("tn"),
        )
        
        # Compute metrics
        perf_stats = perf_stats.withColumn(
            "accuracy",
            (col("tp") + col("tn")) / (col("tp") + col("tn") + col("fp") + col("fn"))
        ).withColumn(
            "precision",
            col("tp") / (col("tp") + col("fp"))
        ).withColumn(
            "recall",
            col("tp") / (col("tp") + col("fn"))
        ).withColumn(
            "f1_score",
            2 * (col("precision") * col("recall")) / (col("precision") + col("recall"))
        )
        
        return perf_stats
    
    def detect_drift_alerts(self, data_drift_df, label_drift_df, 
                            data_threshold=0.1, label_threshold=0.02):
        """
        Detect and flag drift events
        
        Args:
            data_drift_df: Data drift statistics
            label_drift_df: Label drift statistics
            data_threshold: Threshold for data drift detection
            label_threshold: Threshold for label drift (fraud rate change)
        """
        # Flag significant changes
        if data_drift_df:
            data_drift_df = data_drift_df.withColumn(
                "has_data_drift",
                col("stddev_amount") > data_threshold
            )
        
        if label_drift_df:
            label_drift_df = label_drift_df.withColumn(
                "has_label_drift",
                abs(col("fraud_rate_percent") - 0.172) > label_threshold * 100  # 0.172% baseline
            )
        
        return data_drift_df, label_drift_df
    
    def write_monitoring_report(self, data_drift_df, label_drift_df, 
                                pred_drift_df=None, perf_df=None):
        """
        Write monitoring reports to console
        
        Args:
            data_drift_df: Data drift statistics
            label_drift_df: Label drift statistics
            pred_drift_df: Prediction drift statistics
            perf_df: Performance metrics
        """
        queries = []
        
        if data_drift_df is not None:
            q1 = data_drift_df.writeStream \
                .format("console") \
                .option("truncate", "false") \
                .outputMode("update") \
                .start()
            queries.append(q1)
        
        if label_drift_df is not None:
            q2 = label_drift_df.writeStream \
                .format("console") \
                .option("truncate", "false") \
                .outputMode("update") \
                .start()
            queries.append(q2)
        
        if pred_drift_df is not None:
            q3 = pred_drift_df.writeStream \
                .format("console") \
                .option("truncate", "false") \
                .outputMode("update") \
                .start()
            queries.append(q3)
        
        if perf_df is not None:
            q4 = perf_df.writeStream \
                .format("console") \
                .option("truncate", "false") \
                .outputMode("update") \
                .start()
            queries.append(q4)
        
        return queries
    
    def run(self, model_path=None, window_duration="10 minutes"):
        """
        Run the drift monitoring pipeline
        
        Args:
            model_path: Path to pre-trained model
            window_duration: Time window for monitoring
        """
        try:
            # Load model if provided
            if model_path:
                self.load_model(model_path)
            
            # Read from Kafka
            raw_df = self.read_kafka_stream()
            
            # Parse JSON
            parsed_df = self.parse_json_events(raw_df)
            
            # Monitor data drift
            logger.info("Monitoring data drift...")
            data_drift_df = self.monitor_data_drift(parsed_df, window_duration)
            
            # Monitor label drift
            logger.info("Monitoring label drift...")
            label_drift_df = self.monitor_label_drift(parsed_df, window_duration)
            
            # Write reports
            queries = self.write_monitoring_report(data_drift_df, label_drift_df)
            
            # Wait for termination
            for query in queries:
                query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in drift monitoring pipeline: {e}")
            raise
        finally:
            self.spark.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Streaming drift monitoring')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model')
    parser.add_argument('--window', type=str, default='10 minutes',
                        help='Monitoring window duration')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', type=str, default='credit-transactions',
                        help='Kafka topic to consume from')
    
    args = parser.parse_args()
    
    monitor = DriftMonitor(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    
    monitor.run(model_path=args.model, window_duration=args.window)


if __name__ == '__main__':
    main()
