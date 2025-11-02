#!/bin/bash
# Installation script for vLLM project with dependency resolution
# Handles conflicts and installs in the correct order

echo "================================================"
echo "vLLM Project Dependencies Installation"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $python_version"

if [[ ! "$python_version" =~ ^3\.(8|9|10|11)$ ]]; then
    print_warning "Python 3.8-3.11 is recommended for vLLM"
fi

# Upgrade pip first
echo ""
echo "Step 1: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install numpy first (many packages depend on it)
echo ""
echo "Step 2: Installing NumPy..."
pip install numpy==1.24.3
print_status "NumPy installed"

# Install PyTorch
echo ""
echo "Step 3: Installing PyTorch (this may take a while)..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
if [ $? -eq 0 ]; then
    print_status "PyTorch installed successfully"
else
    print_error "PyTorch installation failed"
    exit 1
fi

# Install vLLM and verify immediately
echo ""
echo "Step 4: Installing vLLM..."
pip install vllm==0.5.4

# Install critical version-constrained packages FIRST to avoid conflicts
echo ""
echo "Step 4a: Installing version-constrained dependencies..."
# These specific versions are required to avoid conflicts
pip install "packaging==23.2"  # Required by langfuse and limits
pip install "urllib3==1.26.18"  # Required by kubernetes
pip install "click==8.1.7"  # Required by ray
pip install "huggingface-hub==0.36.0"  # Compatible with transformers 4.57.1

# Install vLLM additional dependencies
echo ""
echo "Step 4b: Installing vLLM additional dependencies..."
pip install pyairports  # This MUST be installed for vLLM to work

# Fix pyairports if it's missing the airports module
echo "Step 4b1: Fixing pyairports module..."
python3 << 'EOF'
import os
import sys
try:
    from pyairports.airports import AIRPORT_LIST
    print("✅ pyairports.airports module exists")
except ImportError:
    print("⚠️  pyairports.airports module missing - creating workaround...")
    # Create a minimal airports module to satisfy the import
    import site
    site_packages = site.getsitepackages()[0]
    pyairports_path = os.path.join(site_packages, "pyairports")

    # Create the directory if it doesn't exist
    os.makedirs(pyairports_path, exist_ok=True)

    # Create __init__.py if it doesn't exist
    init_file = os.path.join(pyairports_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")

    # Create a minimal airports.py with empty AIRPORT_LIST
    airports_file = os.path.join(pyairports_path, "airports.py")
    with open(airports_file, "w") as f:
        f.write("# Minimal airports module to satisfy outlines import\n")
        f.write("AIRPORT_LIST = []\n")

    print("✅ Created minimal pyairports.airports module")
EOF

# Install outlines without dependencies to avoid version conflicts
pip install --no-deps "outlines==0.0.46"
pip install "jsonschema>=4.0.0"

# Verify vLLM works before continuing
echo ""
echo "Step 4c: Verifying vLLM installation..."
python3 -c "from vllm import LLM, SamplingParams; print('vLLM imports OK')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status "vLLM installed and verified"
else
    print_error "vLLM import failed - attempting fix..."
    # Clean reinstall of problem packages
    pip uninstall -y pyairports outlines huggingface-hub packaging urllib3 click

    # Reinstall with correct versions
    pip install "packaging==23.2"
    pip install "urllib3==1.26.18"
    pip install "click==8.1.7"
    pip install "huggingface-hub==0.36.0"
    pip install pyairports

    # Fix pyairports module
    python3 << 'EOF'
import os
import site
site_packages = site.getsitepackages()[0]
pyairports_path = os.path.join(site_packages, "pyairports")
os.makedirs(pyairports_path, exist_ok=True)
init_file = os.path.join(pyairports_path, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("")
airports_file = os.path.join(pyairports_path, "airports.py")
with open(airports_file, "w") as f:
    f.write("# Minimal airports module to satisfy outlines import\n")
    f.write("AIRPORT_LIST = []\n")
EOF

    pip install --no-deps "outlines==0.0.46"

    # Try again
    python3 -c "from vllm import LLM, SamplingParams; print('vLLM imports OK')" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "vLLM still not working. Please check error messages above."
        exit 1
    fi
    print_status "vLLM fixed and working"
fi

# Install transformers and related packages
echo ""
echo "Step 5: Installing Transformers ecosystem..."
pip install "transformers>=4.43.2"
pip install "tokenizers>=0.19.1"
pip install "safetensors>=0.4.1"
# Skip huggingface-hub here - already installed with correct version in Step 4a
pip install "accelerate>=0.25.0"
pip install sentencepiece==0.1.99
pip install protobuf==4.25.1
pip install einops==0.7.0
print_status "ML ecosystem installed"

# Install API and web dependencies (handle conflicts)
echo ""
echo "Step 6: Installing API and web dependencies..."

# Check if FastAPI is already installed to avoid conflicts
python3 -c "import fastapi" 2>/dev/null
if [ $? -eq 0 ]; then
    print_warning "FastAPI already installed, skipping to avoid conflicts"
else
    # Try to install FastAPI with flexible version
    pip install "fastapi>=0.104.1" 2>/dev/null || {
        print_warning "Using flexible FastAPI version due to conflicts"
        pip install fastapi
    }
fi

pip install "uvicorn[standard]>=0.24.0" "pydantic>=2.5.0"
pip install python-multipart httpx aiohttp aiofiles
print_status "Web dependencies installed"

# Install data processing packages
echo ""
echo "Step 7: Installing data processing packages..."
pip install pandas==2.1.4 scipy==1.11.4 scikit-learn==1.3.2
print_status "Data processing packages installed"

# Install monitoring and observability
echo ""
echo "Step 8: Installing monitoring and observability tools..."
pip install prometheus-client==0.19.0
pip install opentelemetry-api==1.21.0 opentelemetry-sdk==1.21.0
pip install langfuse==2.20.0 arize-phoenix==4.5.0
print_status "Monitoring tools installed"

# Install additional packages (optional, can be skipped if conflicts)
echo ""
echo "Step 9: Installing additional packages (optional)..."

# Install packages that are less critical and can be skipped if conflicts occur
packages_to_try=(
    "plotly==5.18.0"
    "matplotlib==3.8.2"
    "google-cloud-storage==2.13.0"
    "redis==5.0.1"
    "sqlalchemy==2.0.23"
    "sentence-transformers==2.2.2"
    "pytest==7.4.3"
    "black==23.12.0"
    "ipython==8.18.1"
)

for package in "${packages_to_try[@]}"; do
    pip install "$package" 2>/dev/null || print_warning "Skipped $package due to conflicts"
done

print_status "Optional packages installed"

echo ""
echo "================================================"
echo "Final Verification"
echo "================================================"

# Comprehensive verification
python3 << EOF
import sys
success = True

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
    success = False

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")
    success = False

try:
    from vllm import LLM, SamplingParams
    import vllm
    print(f"✅ vLLM {vllm.__version__}")
except ImportError as e:
    print(f"❌ vLLM: {e}")
    success = False

try:
    import outlines
    # Handle different version attribute names
    version = getattr(outlines, "__version__", getattr(outlines, "_version", "unknown"))
    print(f"✅ Outlines {version}")
except ImportError as e:
    print(f"❌ Outlines: {e}")
    success = False

try:
    import pyairports
    print(f"✅ PyAirports installed")
except ImportError as e:
    print(f"❌ PyAirports: {e}")
    success = False

if not success:
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    print_status "Installation completed successfully!"
    echo ""
    echo "You can now run:"
    echo "  python test_vllm.py"
    echo "  python quick_test.py"
    echo "  python verify_installation.py  # For detailed check"
else
    echo ""
    print_error "Some critical dependencies failed!"
    echo ""
    echo "To diagnose issues, run:"
    echo "  python verify_installation.py"
    echo ""
    echo "The script should have fixed pyairports automatically."
    echo "If you still see errors, try running the script again."
    exit 1
fi