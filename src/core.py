import streamlit as st
import os
import logging
from src.preprocessing.processor import PromptProcessor
from src.ui.interface import create_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting application")
    
    try:
        processor = PromptProcessor()
        logger.info("PromptProcessor initialized successfully")
        
        # Create and run the Streamlit interface
        create_interface(processor)
        logger.info("Interface created successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
