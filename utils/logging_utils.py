import logging
import os

def setup_logging(output_dir, log_filename):
    """Setup logging configuration
    
    Args:
        output_dir (str): Directory to save log file
        log_filename (str): Name of the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    log_file = os.path.join(output_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Feature Extraction with Stain Normalization Started")
    logger.info("=" * 60)
    
    return logger 