# Granite MoE

IBM's Granite Mixture-of-Experts models for on-device inference via Core AI.

## Supported Models

| Model                | Parameters (Total/Active) | macOS | iOS |
| -------------------- | ------------------------- | ----- | --- |
| Granite 3.1 3b-A800M | 3B / 800M                 | Yes   | Yes |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```
## Export models

```bash
# Defaults to macOS variant
uv run coreai.llm.export ibm-granite/granite-3.1-3b-a800m-instruct
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export ibm-granite/granite-3.1-3b-a800m-instruct --compression none

# Custom output directory
uv run coreai.llm.export ibm-granite/granite-3.1-3b-a800m-instruct --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export ibm-granite/granite-3.1-3b-a800m-instruct --dry-run
```

## Run a Core AI Language Model

### In your iOS and macOS applications via Foundation Models

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAILanguageModel(resourcesAt: modelURL)

let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "What is quantum computing?")

print(response)
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.
