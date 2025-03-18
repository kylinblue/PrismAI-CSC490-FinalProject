import gradio as gr
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
    # Configure for Docker if running in container
    server_name = "0.0.0.0" if os.environ.get("DOCKER_CONTAINER") else None
    server_port = int(os.environ.get("PORT", 7860))
    
    logger.info(f"Server configuration: name={server_name}, port={server_port}")
    
    try:
        processor = PromptProcessor()
        logger.info("PromptProcessor initialized successfully")
        
        interface = create_interface(processor)
        logger.info("Interface created successfully")
        
        interface.launch(server_name=server_name, server_port=server_port)
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
