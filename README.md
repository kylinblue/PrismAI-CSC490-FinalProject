# csc490-finalproject-draft

## Installation

### Using Docker (Recommended)

1. Install Docker and Docker Compose on your system
2. Clone this repository
3. Run the application:
```bash
docker-compose up --build
```

The application will be available at http://localhost:7860

### Manual Installation

1. Ensure Python 3.11+ is installed
2. Run the installation script:
```bash
python scripts/install.py
```

## Usage

The application provides a web interface for interacting with the Mistral language model through Ollama.

## Development

### Running Tests
```bash
pytest tests/
```
