import os
import platform
import subprocess
import sys
from pathlib import Path

def check_docker_installed():
    """Check if Docker is installed."""
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_docker_compose_installed():
    """Check if Docker Compose is installed."""
    try:
        # Docker Compose V2 is integrated into Docker CLI
        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False

def detect_gpu_support():
    """Detect GPU support for Docker."""
    system = platform.system()
    
    # Check for Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        return "apple_silicon"
    
    # Check for NVIDIA GPU on Windows or Linux
    try:
        if system == "Windows":
            # Check for nvidia-smi on Windows
            result = subprocess.run(["where", "nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                return "nvidia"
        else:
            # Check for nvidia-smi on Linux
            result = subprocess.run(["which", "nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                # Verify NVIDIA driver is working
                driver_check = subprocess.run(["nvidia-smi"], capture_output=True)
                if driver_check.returncode == 0:
                    return "nvidia"
    except Exception:
        pass
    
    return "cpu"

def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is installed."""
    try:
        result = subprocess.run(
            ["docker", "info"], 
            capture_output=True, 
            text=True,
            check=True
        )
        return "nvidia" in result.stdout.lower()
    except Exception:
        return False

def install_docker_instructions():
    """Provide instructions for Docker installation."""
    system = platform.system()
    
    print("\n========================================================")
    print("Docker is not installed or not running.")
    print("Please install Docker using these instructions:")
    
    if system == "Windows":
        print("\nWindows:")
        print("1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop")
        print("2. During installation, ensure WSL 2 and 'Use WSL 2 based engine' are selected")
        print("3. If you have NVIDIA GPU, install NVIDIA Container Toolkit after Docker installation")
    elif system == "Darwin":  # macOS
        print("\nmacOS:")
        print("1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop")
        print("2. Follow the installation instructions")
        if platform.machine() == "arm64":
            print("3. Ensure you download the Apple Silicon (M1/M2/M3) version")
    else:  # Linux
        print("\nLinux:")
        print("1. Install Docker using your distribution's package manager")
        print("   or follow: https://docs.docker.com/engine/install/")
        print("2. For NVIDIA GPU support, install NVIDIA Container Toolkit:")
        print("   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
    
    print("\nAfter installation, restart your computer and run this script again.")
    print("========================================================")

def install_nvidia_docker_instructions():
    """Provide instructions for NVIDIA Docker installation."""
    system = platform.system()
    
    print("\n========================================================")
    print("NVIDIA GPU detected, but NVIDIA Docker runtime is not installed.")
    
    if system == "Windows":
        print("\nFor Windows:")
        print("1. Install NVIDIA Container Toolkit for Windows:")
        print("   https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-nvidia-docker")
    else:  # Linux
        print("\nFor Linux:")
        print("1. Install NVIDIA Container Toolkit:")
        print("   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
    
    print("\nAfter installation, restart Docker and run this script again.")
    print("========================================================")

def build_docker_images(gpu_type):
    """Build Docker images based on detected GPU support."""
    print("\nBuilding Docker images...")
    
    # Determine which Dockerfile to use
    if gpu_type == "nvidia":
        print("Building with NVIDIA GPU support...")
        build_cmd = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.nvidia.yml", "build"]
    elif gpu_type == "apple_silicon":
        print("Building for Apple Silicon...")
        build_cmd = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.apple.yml", "build"]
    else:
        print("Building for CPU only...")
        build_cmd = ["docker", "compose", "build"]
    
    try:
        subprocess.run(build_cmd, check=True)
        print("✓ Docker images built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build Docker images: {str(e)}")
        return False

def main():
    """Main Docker installation routine."""
    print("\n=== Docker Setup Process ===\n")
    
    try:
        # Check Docker installation
        print("Checking Docker installation...")
        if not check_docker_installed():
            install_docker_instructions()
            return 1
        print("✓ Docker is installed")
        
        # Check Docker Compose
        print("\nChecking Docker Compose...")
        if not check_docker_compose_installed():
            print("❌ Docker Compose is not available")
            print("Please ensure Docker Desktop is running or Docker Compose is installed")
            return 1
        print("✓ Docker Compose is available")
        
        # Detect GPU support
        print("\nDetecting GPU support...")
        gpu_type = detect_gpu_support()
        if gpu_type == "nvidia":
            print("✓ NVIDIA GPU detected")
            
            # Check NVIDIA Docker runtime
            print("\nChecking NVIDIA Docker runtime...")
            if not check_nvidia_docker():
                install_nvidia_docker_instructions()
                return 1
            print("✓ NVIDIA Docker runtime is available")
        elif gpu_type == "apple_silicon":
            print("✓ Apple Silicon detected")
        else:
            print("No GPU detected, using CPU mode")
        
        # Build Docker images
        if not build_docker_images(gpu_type):
            return 1
        
        print("\n=== Docker Setup Summary ===")
        print("✓ Docker environment is ready")
        
        # Print run instructions
        print("\nTo start the application:")
        if gpu_type == "nvidia":
            print("    docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up")
        elif gpu_type == "apple_silicon":
            print("    docker compose -f docker-compose.yml -f docker-compose.apple.yml up")
        else:
            print("    docker compose up")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during Docker setup: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
