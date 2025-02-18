import gradio as gr
import os
from src.preprocessing.processor import PromptProcessor
from src.ui.interface import create_interface

def main():
    # Configure for Docker if running in container
    server_name = "0.0.0.0" if os.environ.get("DOCKER_CONTAINER") else None
    server_port = int(os.environ.get("PORT", 7860))
    
    processor = PromptProcessor()
    interface = create_interface(processor)
    interface.launch(server_name=server_name, server_port=server_port)

if __name__ == "__main__":
    main()
