#Write your schema inference code here

import json
import argparse
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, schema_of_json, udf, current_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
import joblib
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingInference:
    """
    Real-time fraud detection using Spark Structured Streaming.
    
    Workflow:
    1. Read transactions from Kafka
    2. Parse JSON events
    3. Load pre-trained ML model
    4. Apply model to each event
    5. Output predictions with confidence scores
    """
    
    def __init__(self, app_name="StreamingInference", 
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
        self.model = None
        
        logger.info("Streaming Inference initialized")
    
    def load_model(self, model_path):
        """
        Load pre-trained scikit-learn model
        
        Args:
            model_path: Path to the saved model (.pkl or .joblib file)
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
    
    def train_simple_model(self, csv_path):
        """
        Train a simple random forest model for fraud detection.
        Used if no pre-trained model is available.
        
        Args:
            csv_path: Path to credit card CSV file
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Training a simple Random Forest model...")
            
            # Load data
            df = pd.read_csv(csv_path)
            
            # Prepare features and target
            X = df.drop('Class', axis=1) if 'Class' in df.columns else df
            y = df['Class'] if 'Class' in df.columns else None
            
            if y is None:
                logger.warning("No 'Class' column found. Using mock labels.")
                y = [0] * len(X)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
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
        # Parse the JSON value
        parsed_df = df.select(
            from_json(col("value").cast("string"), self._get_schema()).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def _get_schema(self):
        """
        Define schema for credit card transaction events.
        Adjust based on your actual data structure.
        """
        # Example schema - modify to match your CSV columns
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
    
    def apply_model(self, df):
        """
        Apply pre-trained model to streaming data
        
        Args:
            df: Parsed Spark DataFrame with transaction features
            
        Returns:
            DataFrame with predictions
        """
        # Define UDF for model prediction
        def predict(features):
            try:
                # Convert to the format expected by the model
                import numpy as np
                features_array = np.array(features).reshape(1, -1)
                prediction = self.model.predict(features_array)[0]
                probability = self.model.predict_proba(features_array)[0][1]
                return float(prediction), float(probability)
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                return None, None
        
        # Define feature columns (adjust based on your model)
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        
        # Create prediction UDF
        prediction_udf = udf(
            lambda *cols: predict(list(cols)),
            "struct<prediction:double,probability:double>"
        )
        
        # Apply model
        predictions = df.select(
            "*",
            prediction_udf(*[col(c) for c in feature_cols]).alias("fraud_prediction")
        ).select(
            "timestamp",
            "Amount",
            col("fraud_prediction.prediction").alias("is_fraud"),
            col("fraud_prediction.probability").alias("fraud_probability")
        )
        
        return predictions
    
    def write_output(self, df, output_mode="append"):
        """
        Write predictions to console and/or files
        
        Args:
            df: DataFrame with predictions
            output_mode: Spark streaming output mode (append, update, complete)
        """
        query = df.writeStream \
            .format("console") \
            .option("truncate", "false") \
            .outputMode(output_mode) \
            .start()
        
        return query
    
    def run(self, model_path=None, csv_path='creditcard.csv'):
        """
        Run the streaming inference pipeline
        
        Args:
            model_path: Path to pre-trained model
            csv_path: Path to CSV for training a simple model if needed
        """
        try:
            # Load or train model
            if model_path:
                self.load_model(model_path)
            else:
                logger.info("No model provided. Training a simple model...")
                self.train_simple_model(csv_path)
            
            # Read from Kafka
            raw_df = self.read_kafka_stream()
            
            # Parse JSON
            parsed_df = self.parse_json_events(raw_df)
            
            # Apply model
            predictions_df = self.apply_model(parsed_df)
            
            # Write output
            query = self.write_output(predictions_df)
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            raise
        finally:
            self.spark.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Streaming fraud detection inference')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model')
    parser.add_argument('--csv', type=str, default='creditcard.csv',
                        help='Path to credit card CSV file')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', type=str, default='credit-transactions',
                        help='Kafka topic to consume from')
    
    args = parser.parse_args()
    
    inference = StreamingInference(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    
    inference.run(model_path=args.model, csv_path=args.csv)


if __name__ == '__main__':
    main()
