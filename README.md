# Adam
### Autonomously Jailbreaking LLMs via Web Interfaces using Local Models

---

## Overview
**Adam** is a framework for autonomously jailbreaking large language models through real-world web interfaces.
It uses locally hosted models to simulate an **attacker–judge loop**, enabling scalable AI red-teaming experiments.

---

## Features
- Autonomous attacker LLM generating jailbreak prompts
- Judge LLM evaluating success/failure
- Works through real web interfaces (browser-based targets)
- Fully local (via Ollama)
- Web-based UI for monitoring and control
- Multiple attack strategies supported
- Configurable targets and goals

---

## Prerequisites

### System Requirements
- **Python**: 3.12 or higher
- **Node.js**: 18+ (for frontend development)
- **Ollama**: Local LLM runtime (install from [ollama.ai](https://ollama.ai))

### Hardware Requirements
- At least 16GB RAM recommended
- GPU with CUDA support for better performance (optional but recommended)

---

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd adam
```

### 2. Install Python Dependencies
```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Install Ollama and Pull Required Models
```bash
# Install Ollama (follow instructions at https://ollama.ai)
# Then pull the required models:
ollama pull huihui_ai/qwen2.5-abliterate:7b-instruct
ollama pull huihui_ai/deepseek-r1-abliterated:8b
```

> **Note**: These models are specialized for red teaming tasks. The first is used as the judge/evaluator, the second as the attacker.

### 4. Setup Frontend (Optional - for web interface)
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## Configuration

### Core Configuration
Edit `core.py` to modify:

- **TARGET_URL**: The web interface URL to target (default: DeepAI chat)
- **GOAL**: The objective for the jailbreak attempts
- **ITERATIONS**: Number of attack iterations to run
- **LOGIN_EMAIL/LOGIN_PASSWORD**: Credentials if the target requires authentication

### Attack Parameters
Additional configuration options in `Config` class:
- `PROMPTS_PER_STATE`: Number of prompts to generate per reasoning state
- `BUFFER_SIZE`: Size of the prompt buffer
- `TOP_K`: Number of top prompts to select
- `REQUEST_TIMEOUT`: Timeout for web requests
- `PROMPT_DELAY`: Delay between sending prompts

---

## Model Roles

| Role      | Model |
|----------|------|
| Judge    | `huihui_ai/qwen2.5-abliterate:7b-instruct` |
| Attacker | `huihui_ai/deepseek-r1-abliterated:8b` |

---

## Running the Project

### Start the Web Server
```bash
# Activate virtual environment (if not using uv run)
source .venv/bin/activate

# Start the FastAPI server
uv run python server.py
```
The server will start on `http://localhost:3000`

### Run Attacks Directly
```bash
# Run with default adversarial reasoning attack
uv run python main.py

# Run specific attack type
uv run python main.py --attack-type gcg
```

### Available Attack Types
- `adversarial_reasoning` (default)
- `gcg` (Greedy Coordinate Gradient)
- `adaptive_random_search`
- `pair` (Prompt Automatic Iterative Refinement)
- `tap` (Tree of Attacks with Pruning)
- `dan` (Do Anything Now)
- `many_shot` (Many-Shot Jailbreaking)
- `autodan` (AutoDAN)
- `multi_turn` (Multi-turn Jailbreaking)
- `poisonedrag_router`
- `phantom_router`
- `imprompter_toolcall`
- `jailbreakfunction_toolcall`
- `agentharm_iterator`
- `robopair_iterator`

### Frontend Development
```bash
cd frontend
npm run dev
```
Frontend will be available at `http://localhost:5173` (Vite dev server)

---

## How It Works
1. The **attacker model** generates adversarial prompts designed to bypass safety filters
2. Prompts are automatically sent to the target LLM via its web interface using Playwright
3. The **judge model** evaluates the success of each attempt based on the response
4. Results are logged and can be iterated upon for improved attacks
5. The web interface allows monitoring progress and adjusting parameters in real-time

---

## Logs and Output
- Logs are saved to `logs/` directory with timestamps
- Each run creates a new log file: `logs/adversarial_reasoning_YYYYMMDD_HHMMSS.log`
- Global memory is stored in `logs/global_memory.json`
- Screenshots are saved as `screenshot_viewport.png`

---

## Troubleshooting

### Common Issues
- **Ollama not responding**: Ensure Ollama is running with `ollama serve`
- **Browser automation fails**: Check that Playwright browsers are installed: `playwright install`
- **Target website blocks automation**: Some sites detect and block automated browsers
- **Memory issues**: Reduce `BUFFER_SIZE` or `PROMPTS_PER_STATE` if running out of RAM

### Performance Tips
- Use GPU-enabled Ollama for faster inference
- Reduce `ITERATIONS` for quicker testing
- Monitor system resources during long runs

---

## Development

### Project Structure
```
adam/
├── core.py                 # Core utilities and configuration
├── main.py                 # Main attack execution script
├── server.py               # FastAPI web server
├── attacks/                # Attack strategy implementations
├── datasets/               # Training data and attack patterns
├── frontend/               # React web interface
├── logs/                   # Generated logs and memory files
└── scripts/                # Utility scripts
```

### Adding New Attacks
1. Create a new attack class in `attacks/` directory
2. Implement the `run()` method
3. Add the attack to the `attacks` dictionary in `main.py`

---

## Notes
- Designed for experimentation and research in AI safety / red teaming
- Currently does **not** support decomposition-based attacks
- Requires Ollama running locally
- All operations are performed locally - no data is sent to external services

---

## Disclaimer
This project is intended for **educational and research purposes only**.
Do not use it against systems without proper authorization.
The authors are not responsible for any misuse or damage caused by this tool.

---

## Future Work
- Decomposition attacks support
- Memory-based attack improvement
- Multi-model evaluation
- Better success scoring heuristics
- Support for additional target platforms

---
