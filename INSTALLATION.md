# Installation Guide

## Install from Git Source (Recommended)

This package is designed to be installed from GitHub source for both `rl-cua` and `OSWorld-provider` projects.

### For rl-cua Project

Add to `requirements.txt`:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then install:
```bash
cd /path/to/rl-cua
pip install -r requirements.txt
```

### For OSWorld-provider Project

Add to `requirements.txt`:
```
git+https://github.com/babelcloud/gbox-cua.git
```

Then install:
```bash
cd /path/to/OSWorld-provider
pip install -r requirements.txt
```

### Direct Installation

You can also install directly:
```bash
pip install git+https://github.com/babelcloud/gbox-cua.git
```

### Development Installation

If you want to develop the package locally:

```bash
# Clone the repository
git clone https://github.com/babelcloud/gbox-cua.git
cd gbox-cua

# Install in development mode
pip install -e .
```

## Verification

After installation, verify it works:

```python
python -c "from gbox_cua import get_tools_schema; print('Installation successful!')"
```

## Usage

After installation, import in your code:

```python
from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator
```

## Updating

To update to the latest version:

```bash
pip install --upgrade git+https://github.com/babelcloud/gbox-cua.git
```

Or if installed via requirements.txt:

```bash
pip install --upgrade -r requirements.txt
```

