# Installation Guide

## Option 1: Development Installation (Recommended)

Install the package in development mode so changes are immediately available:

```bash
cd /path/to/gbox-cua-agent
pip install -e .
```

## Option 2: Add to PYTHONPATH

If you prefer not to install the package, you can add it to your PYTHONPATH:

```bash
export PYTHONPATH=/path/to/gbox-cua-agent:$PYTHONPATH
```

## Option 3: Install from Local Path

You can also install it directly from the local path in your project's requirements:

```bash
pip install -e /path/to/gbox-cua-agent
```

## Usage in Projects

After installation, both `rl-cua` and `OSWorld-provider` projects can import:

```python
from gbox_cua_agent.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua_agent.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua_agent.gbox_coordinate import GBoxCoordinateGenerator
```

## Project Setup

### For rl-cua project:

```bash
cd /path/to/rl-cua
pip install -e ../gbox-cua-agent
pip install -r requirements.txt
```

### For OSWorld-provider project:

```bash
cd /path/to/OSWorld-provider
pip install -e ../gbox-cua-agent
pip install -r requirements.txt
```

## Docker Setup

### Build Docker Image

```bash
cd /path/to/gbox-cua-agent
docker build -t gbox-cua-agent:latest .
```

### Run in Docker

```bash
docker run -e GBOX_API_KEY=your_key \
           -e VLLM_API_BASE=http://localhost:8000/v1 \
           gbox-cua-agent:latest \
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
pip install -e /path/to/gbox-cua-agent
pip install -e /path/to/rl-cua

# Run agent
gbox-cua-agent "Open the Settings app" --verbose
```

## Verification

Test the installation:

```python
python -c "from gbox_cua_agent import get_tools_schema; print('Installation successful!')"
```

Or test the CLI:

```bash
gbox-cua-agent --help
```
