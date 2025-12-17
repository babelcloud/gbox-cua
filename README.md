# GBox CUA Agent

GBox CUA Agent is a flexible package that can be used in two ways:
1. **Standalone Agent** - Run directly or in Docker for computer use tasks
2. **Library** - Imported by other projects (rl-cua, OSWorld-provider, etc.)

## Features

- **Tool Definitions**: perform_action, sleep, report_task_complete
- **Prompt Templates**: System prompts and user message templates
- **Coordinate Generation**: GBox model integration for precise UI element targeting
- **Standalone CLI**: Run agent directly from command line
- **Docker Support**: Containerized deployment ready

## Installation

### Install from Git Source (Recommended)

```bash
pip install git+https://github.com/babelcloud/gbox-cua.git
```

### Development Installation

```bash
git clone https://github.com/babelcloud/gbox-cua.git
cd gbox-cua
pip install -e .
```

### As a Standalone Agent

`gbox-cua` can run independently without requiring rl-cua or OSWorld:

```bash
# Install gbox-cua from git
pip install git+https://github.com/babelcloud/gbox-cua.git

# Set environment variables
export GBOX_API_KEY="your_gbox_api_key"
export VLM_PROVIDER="openai"  # or "vllm" or "openrouter"
export OPENAI_API_KEY="your_openai_key"  # if using OpenAI

# Run agent
gbox-cua "Open the Settings app"
```

The standalone agent supports:
- **VLM Providers**: vllm, openai, openrouter
- **Box Types**: android, linux
- **All standard actions**: click, swipe, scroll, input, key_press, button_press

## Usage

### As a Library

```python
from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator

# Use in your agent implementation
tools = get_tools_schema()
system_prompt = create_system_prompt("Your task description", max_turns=20)
```

### As a Standalone Agent

```bash
# Set environment variables
export GBOX_API_KEY="your_gbox_api_key"

# Option 1: Using OpenAI
export VLM_PROVIDER="openai"
export OPENAI_API_KEY="your_openai_key"
gbox-cua "Open the Settings app"

# Option 2: Using OpenRouter
export VLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="your_openrouter_key"
gbox-cua --task "Search for weather" --box-type android --verbose

# Option 3: Using vLLM (local server)
export VLM_PROVIDER="vllm"
export VLLM_API_BASE="http://localhost:8000/v1"
gbox-cua --task "Navigate to gmail.com" --box-type linux
```

### Using Docker

#### Quick Start

```bash
# 1. Configure environment variables
cp env.example .env
# Edit .env and fill in your API keys

# 2. Build Docker image
./build_docker.sh

# 3. Run agent
./docker_run.sh "Open the Settings app"
```

#### Manual Docker Commands

```bash
# Build image
docker build -t gbox-cua:latest .

# Run with environment variables
docker run -e GBOX_API_KEY=your_key \
           -e VLLM_API_BASE=http://localhost:8000/v1 \
           gbox-cua:latest \
           "Open the Settings app"

# Or use docker-compose
docker-compose up
```

#### Docker Files

- `Dockerfile`: Docker image definition
- `build_docker.sh`: Script to build the Docker image
- `docker_run.sh`: Script to run the agent in Docker
- `docker-compose.yml`: Docker Compose configuration
- `env.example`: Environment variables template

## Package Contents

- `tools.py`: Tool definitions (perform_action, sleep, report_task_complete)
- `prompts.py`: Prompt templates for the agent
- `gbox_coordinate.py`: GBox coordinate generation logic
- `cli.py`: Command-line interface for standalone usage

## Integration with Other Projects

### rl-cua

Add to requirements.txt:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then import:
```python
from gbox_cua.tools import get_tools_schema
from gbox_cua.prompts import create_system_prompt
```

### OSWorld-provider

Add to requirements.txt:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then import:
```python
from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
```

## Requirements

- Python >= 3.8
- gbox-sdk
- httpx >= 0.25.0
- Pillow >= 10.0.0

## License

MIT License
