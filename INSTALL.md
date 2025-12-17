# Installation Guide

## Install from Git Source (Recommended)

```bash
pip install git+https://github.com/babelcloud/gbox-cua.git
```

## Development Installation

```bash
git clone https://github.com/babelcloud/gbox-cua.git
cd gbox-cua
pip install -e .
```

## Usage in Projects

After installation, both `rl-cua` and `OSWorld-provider` projects can import:

```python
from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator
```

## Project Setup

### For rl-cua project:

Add to `requirements.txt`:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then install:
```bash
cd /path/to/rl-cua
pip install -r requirements.txt
```

### For OSWorld-provider project:

Add to `requirements.txt`:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then install:
```bash
cd /path/to/OSWorld-provider
pip install -r requirements.txt
```

## Docker Setup

### Build Docker Image

```bash
git clone https://github.com/babelcloud/gbox-cua.git
cd gbox-cua
docker build -t gbox-cua:latest .
```

### Run in Docker

```bash
docker run -e GBOX_API_KEY=your_key \
           -e VLLM_API_BASE=http://localhost:8000/v1 \
           gbox-cua:latest \
           "Open the Settings app"
```

### Using docker-compose

```bash
# Set environment variables in .env file
echo "GBOX_API_KEY=your_key" > .env
echo "VLLM_API_BASE=http://localhost:8000/v1" >> .env
echo "TASK=Open the Settings app" >> .env

# Run
docker-compose up
```

## Standalone Agent Usage

For full standalone agent functionality, you need rl-cua installed:

```bash
# Install both packages
pip install git+https://github.com/babelcloud/gbox-cua.git
pip install -e /path/to/rl-cua

# Run agent
gbox-cua "Open the Settings app" --verbose
```

## Verification

Test the installation:

```python
python -c "from gbox_cua import get_tools_schema; print('Installation successful!')"
```

Or test the CLI:

```bash
gbox-cua --help
```
