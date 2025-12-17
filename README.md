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

For full standalone functionality, you'll also need rl-cua:

```bash
# Install gbox-cua from git
pip install git+https://github.com/babelcloud/gbox-cua.git

# Install rl-cua for full agent implementation
cd /path/to/rl-cua
pip install -e .
```

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
export VLLM_API_BASE="http://localhost:8000/v1"  # Optional

# Run agent
gbox-cua "Open the Settings app"
gbox-cua --task "Search for weather" --box-type android --verbose
```

### Using Docker

```bash
# Build image
docker build -t gbox-cua-agent:latest .

# Run with environment variables
docker run -e GBOX_API_KEY=your_key \
           -e VLLM_API_BASE=http://localhost:8000/v1 \
           gbox-cua-agent:latest \
           "Open the Settings app"

# Or use docker-compose
docker-compose up
```

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
