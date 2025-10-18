# Mobile GUI Agent Framework

A mobile GUI agent framework that uses reasoning-enabled Vision-Language Models (VLMs) for automated smartphone control, based on the paper "Does Chain-of-Thought Reasoning Help Mobile GUI Agent? An Empirical Study".

## Features

- Support for reasoning-enabled VLMs (Claude 3.7 Sonnet, Gemini 2.0 Flash, GPT-4)
- Multiple prompting strategies (Set-of-Mark, Accessibility Tree)
- GUI element grounding with coordinate prediction
- Action execution for mobile automation
- Chain-of-thought reasoning for complex tasks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from mobile_gui_agent import MobileGUIAgent
from mobile_gui_agent.vlm import ClaudeVLM

# Initialize agent with reasoning-enabled VLM
agent = MobileGUIAgent(
    vlm=ClaudeVLM(model="claude-3-7-sonnet", use_reasoning=True),
    prompting_strategy="set_of_mark"
)

# Execute a task
result = agent.execute_task(
    screenshot_path="screenshot.png",
    task_instruction="Set my DM Spam filter to 'Do not filter direct messages' on Discord app"
)

print(f"Action: {result['action']}")
print(f"Reasoning: {result['reasoning']}")
```

## Architecture

- `mobile_gui_agent/`: Core framework
  - `agent.py`: Main agent implementation
  - `vlm/`: VLM provider implementations
  - `grounding.py`: GUI element grounding
  - `actions.py`: Action execution
  - `prompting.py`: Prompting strategies

## Supported VLMs

- Claude 3.7 Sonnet (with/without reasoning)
- Gemini 2.0 Flash (with/without reasoning)
- GPT-4o

## License

MIT
