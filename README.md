# LLM Alignment Assistant

## Project Overview

The LLM Alignment Assistant is a wrapper application that helps users align language models to produce text with their intended style, tone, and content requirements. This tool simplifies the process of generating text with user-intended style or tone by providing an intuitive interface for prompt engineering and model alignment.

### Key Benefits

- **Reduced Barriers**: Lowers the barrier to entry for producing aligned and uncensored text-to-text results
- **On-the-fly Fine-tuning**: Provides "on the fly" LLM fine-tuning and alignment capabilities
- **Resource Integration**: Allows providing relevant resources such as text or images to improve model output

## Installation

### Using Installation Scripts (Recommended)

#### Windows
1. Ensure Python 3.11+ is installed
2. **Important**: If you plan to use local inference, [install Ollama](https://ollama.ai/download) manually first
3. Run the installation batch file:
```bash
install.bat
```

#### macOS/Linux
1. Ensure Python 3.11+ is installed
2. Make the installation script executable and run it:
```bash
chmod +x install.sh
./install.sh
```

These scripts will:
- Check your Python version
- Create a virtual environment
- Install required dependencies
- On macOS/Linux: Attempt to automatically install Ollama if not present
- On Windows: Guide you to install Ollama manually if needed
- Pull necessary language models

After installation, activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

### Manual Installation

If you prefer to install manually:

1. Ensure Python 3.11+ is installed
2. If you plan to use local inference, install Ollama:
   - Windows: Download from [ollama.ai/download](https://ollama.ai/download)
   - macOS/Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
3. Run the Python installation script directly:
```bash
python scripts/install.py
```

### Docker Installation (Coming Soon)

Docker support is under development and will be available in a future release.

## Usage

The application provides a web interface for interacting with language models through various inference engines:

1. Start the application:
```bash
# Activate virtual environment first
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Then run the application
python -m src.core
```

2. Access the web interface at http://localhost:7860
3. Enter your prompt in the input field
4. Adjust alignment parameters as needed
5. Submit your request to generate aligned text
6. Review and refine the output as necessary

**Note**: For local inference, ensure Ollama is running in the background. The application will attempt to connect to Ollama's API at http://localhost:11434.

## Features

- **Multiple Engine Support**: Works with Ollama, Claude, and other LLM backends
- **Style and Tone Control**: Generate text that matches your intended style or tone
- **User-Friendly Interface**: Simple web interface inspired by [Ollama GUI](https://ollama-gui.vercel.app)

## Development

### Project Structure

- `src/core.py`: Core application functionality
- `src/inferencing/`: Inference engine implementations
- `src/preprocessing/`: Prompt processing and alignment logic
- `src/ui/`: Web interface components
- `tests/`: Test suite

### Running Tests
```bash
pytest tests/
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
