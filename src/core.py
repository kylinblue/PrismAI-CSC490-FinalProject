import gradio as gr
from src.preprocessing.processor import PromptProcessor
from src.ui.interface import create_interface

def main():
    processor = PromptProcessor()
    interface = create_interface(processor)
    interface.launch()

if __name__ == "__main__":
    main()