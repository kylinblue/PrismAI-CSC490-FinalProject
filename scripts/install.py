import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        print("\n========================================================")
        print("ERROR: Python 3.8 or higher is required")
        print("Current version:", sys.version.split()[0])
        print("\nPlease download and install a newer version from:")
        print("https://www.python.org/downloads/")
        print("========================================================")
        sys.exit(1)

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(".venv", with_pip=True)
        return True
    return False

def install_requirements():
    """Install Python package requirements."""
    pip_cmd = ".venv/bin/pip" if platform.system() != "Windows" else r".venv\Scripts\pip"
    subprocess.check_call([pip_cmd, "install", "-r", "requirements.txt"])

def install_ollama():
    """Install Ollama if not present."""
    if platform.system() == "Windows":
        # On Windows, we can only guide the user
        result = subprocess.run(["where", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("\n========================================================")
            print("Ollama installation required:")
            print("1. Please visit https://ollama.ai/download")
            print("2. Download and run the Windows installer")
            print("3. After installation, restart this script")
            print("========================================================")
            return False
    else:
        # On Unix-like systems, we can attempt automatic installation
        result = subprocess.run(["which", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("\nInstalling Ollama...")
            try:
                # Download and run the Ollama install script
                subprocess.run(
                    ["curl", "-fsSL", "https://ollama.ai/install.sh"],
                    stdout=subprocess.PIPE,
                    check=True
                ).stdout | subprocess.run(
                    ["sh"],
                    check=True
                )
                print("Ollama installed successfully!")
            except subprocess.CalledProcessError:
                print("\n========================================================")
                print("Automatic Ollama installation failed.")
                print("Please install manually with:")
                print("    curl https://ollama.ai/install.sh | sh")
                print("========================================================")
                return False
    
    # Pull the Mistral model
    print("\nPulling Mistral model (this may take a while)...")
    try:
        subprocess.run(["ollama", "pull", "mistral"], check=True)
        print("Mistral model downloaded successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to pull Mistral model. Please run manually:")
        print("    ollama pull mistral")
        return False

def main():
    """Main installation routine."""
    print("Starting installation process...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    venv_created = create_venv()
    
    # Install requirements
    print("Installing requirements...")
    install_requirements()
    
    # Install Ollama and pull model
    ollama_ready = install_ollama()
    
    print("\nInstallation completed!")
    if venv_created:
        print("\nVirtual environment created. To activate:")
        if platform.system() == "Windows":
            print("    .venv\\Scripts\\activate")
        else:
            print("    source .venv/bin/activate")
    
    if not ollama_ready:
        print("\nNOTE: Please complete Ollama setup using the instructions above before proceeding.")
    
    print("\nSetup complete! You can now run the application.")

if __name__ == "__main__":
    main()
