import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 11):
        print("\n========================================================")
        print("ERROR: Python 3.11 or higher is required")
        print("Current version:", sys.version.split()[0])
        print("\nPlease download and install a newer version from:")
        print("https://www.python.org/downloads/")
        print("========================================================")
        sys.exit(1)

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_name = ".venv"
    venv_path = Path(venv_name)
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_name, with_pip=True)
        return True
    return False

def install_requirements():
    """Install Python package requirements."""
    venv_name = ".venv"
    pip_cmd = f"{venv_name}/bin/pip" if platform.system() != "Windows" else fr"{venv_name}\Scripts\pip"
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
    print("\n=== Starting Installation Process ===\n")
    
    try:
        # Check Python version
        print("Checking Python version...")
        check_python_version()
        print("✓ Python version OK")
        
        # Create virtual environment
        print("\nSetting up virtual environment...")
        venv_created = create_venv()
        if venv_created:
            print("✓ Virtual environment created")
        else:
            print("✓ Using existing virtual environment")
        
        # Install requirements
        print("\nInstalling Python dependencies...")
        install_requirements()
        print("✓ Dependencies installed")
        
        # Install Ollama and pull model
        print("\nChecking Ollama installation...")
        ollama_ready = install_ollama()
        
        print("\n=== Installation Summary ===")
        print("✓ Python environment setup complete")
        if venv_created:
            venv_name = ".venv"
            print("\nTo activate virtual environment:")
            if platform.system() == "Windows":
                print(f"    {venv_name}\\Scripts\\activate")
            else:
                print(f"    source {venv_name}/bin/activate")
        
        if not ollama_ready:
            print("\n⚠ Action Required:")
            print("Please complete Ollama setup using the instructions above")
            print("before running the application.")
            return 1
        
        print("\n✓ Setup complete! You can now run the application.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during installation: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    main()
