from customer_behaviour_bi_lstm import CustomerBehaviorPipeline
#from customer_behaviour_bi_lstm_week import CustomerBehaviorPipeline
import logging

customer = 703538803 # Change customer refrence as wished

# Configure logging for the test
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to capture all log messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_single_customer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Initialize the pipeline
        pipeline = CustomerBehaviorPipeline()
        
        # Connect to the database
        logger.info("Connecting to database...")
        pipeline.connect_db()
        
        # Process a single customer (replace 1 with your customer_ref)
        customer_ref = customer
        logger.info(f"Processing customer {customer_ref}")
        prediction, plot_path, json_path = pipeline.process_customer(customer_ref=customer_ref, sequence_length=96, batch_size=32)
        
        # Log the results
        if prediction is not None and plot_path is not None and json_path is not None:
            logger.info(f"Customer {customer_ref}: Prediction: {prediction:.4f} kW, Plot: {plot_path}, JSON: {json_path}")
        else:
            logger.error(f"Failed to process customer {customer_ref}: No valid output generated")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    finally:
        # Close the database connection
        logger.info("Closing database connection...")
        pipeline.close_db()