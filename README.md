# ScaleDistill

ScaleDistill is an experimental method for correcting packed low-bit language models while preserving their deployable quantized structure.

Instead of restoring and fine-tuning dense weights, ScaleDistill freezes the packed integer codes and optimizes the continuous quantization scales used to reconstruct them. The resulting checkpoint can be merged directly into the original safetensors shards without changing the model architecture or packed weight format.

Status: private research project under active development. Results and APIs are not yet stable.

## Motivation

Very low-bit quantization can preserve model size and inference efficiency but distort output distributions, reasoning, code generation, and tool use. Re-quantizing from a dense model is expensive, and the original dense weights may no longer be available.

ScaleDistill asks a narrower question:

> Given frozen packed integer codes, what scale values make the quantized model behave as closely as possible to a stronger teacher and the desired data?

This makes it possible to improve an existing packed artifact using teacher logits, next-token targets, or assistant-only behavioral supervision.

## Core Model

For a quantized weight group:

```text
W_i = q_i * s
```

Where:

- `q_i` is a frozen low-bit integer code.
- `s` is a trainable scale shared by a quantization group.
- `W_i` is the reconstructed weight used by the linear layer.

For example:

```text
codes:         [-2, -1, 0, 1]
initial scale: 0.46
weights:       [-0.92, -0.46, 0.00, 0.46]

learned scale: 0.50
weights:       [-1.00, -0.50, 0.00, 0.50]
```

The integer pattern stays fixed. Training learns a better magnitude for each group.

## Training Workflow

1. Load the packed quantized artifact.
2. Locate `weight_packed`, `weight_shape`, and `weight_scale` tensors.
3. Freeze all packed integer codes.
4. Replace inference-only dequantization with a differentiable fake-quant forward pass.
5. Unpack integer codes without gradients.
6. Multiply frozen codes by trainable scales.
7. Compute a teacher-KL, next-token CE, or assistant-masked behavior loss.
8. Backpropagate into deployable scale parameters.
9. Save independent best-PPL and best-behavior checkpoints.
10. Merge selected scales into a copy of the packed safetensors artifact.
11. Re-evaluate the merged model under normal inference loading.

## Differentiable Packed Forward

Compressed inference kernels are designed for serving and may not expose an autograd path into their scale tensors. ScaleDistill reconstructs the forward operation explicitly:

```python
codes = unpack_from_int32(weight_packed)
scale = weight_scale.repeat_interleave(group_size, dim=1)
weight = codes * scale
output = linear(hidden_states, weight, bias)
```

The unpacked codes are constants. Gradients flow only through the scales and any additional explicitly enabled deployable parameters.

## Objectives

### Teacher KL

A teacher pass stores each sequence's input IDs and top-k token distribution. The student minimizes KL divergence against the cached teacher distribution. The teacher does not remain in memory during student training.

### Direct Language Modeling CE

Next-token cross-entropy directly optimizes token likelihood and perplexity. An independent evaluation text is never used for gradient updates and is used only for checkpoint selection.

### Assistant-Masked Behavior CE

Behavioral training supervises assistant responses and tool actions while masking system prompts, user messages, and tool outputs with `-100` labels.

```text
System context:  ignored
User request:    ignored
Tool output:     ignored
Assistant turn:  supervised
```

This focuses updates on model behavior rather than prompt reproduction.

### Behavioral Curriculum

The current balanced pilot corpus is built from private datasets:

- `wannabecomputerscientist/pi-office-levels-v2`
- `wannabecomputerscientist/pi-office-open-swe-fraction`

The pilot contains:

| Statistic | Value |
| --- | --- |
| Training examples | 544 |
| Validation examples | 111 |
| Supervised target tokens | 151,950 |

Coverage includes exact arithmetic, fraction reasoning, solver routing, coding, debugging, OpenHands trajectories, SWEAgent trajectories, and tool calls. Failed SWE trajectories are excluded from positive supervision.

## Memory Strategy

Training a 27B packed model still requires careful memory management:

- Decoder-layer gradient checkpointing discards large intermediate tensors.
- Adafactor provides factored optimizer state for two-dimensional scale matrices.
- Full-vocabulary logits are processed in position chunks.
- Scale checkpoints are saved independently from model shards.
- Candidate artifacts are merged and benchmarked one at a time.

This allows the current W2A16-G32 experiment to run on one A100 80GB GPU.

## Checkpoint Selection

The final training step is not assumed to be best. ScaleDistill tracks separate checkpoints for:

- Lowest independent perplexity
- Lowest masked behavior-validation CE
- Final training state

This preserves useful early and middle checkpoints if later updates overfit or trade general language quality for task behavior.

## Current Experimental Results

Independent merged-artifact measurements for W2A16-G32:

| Metric | Original G32 | ScaleDistill |
| --- | ---: | ---: |
| Teacher KL, 16 samples | 0.310772 | 0.042939 |
| Natural-text perplexity | 1.744811 | 1.736018 |

The current behavior run is still in progress. Its checkpoints will be tested against the original model, initial ScaleDistill model, best-PPL state, best-behavior state, and final state.

## Benchmark Matrix

Each merged candidate is evaluated on:

- Natural-text perplexity
- Teacher KL
- Assistant-masked behavioral CE
- Exact integer arithmetic
- Exact fraction arithmetic
- Python code generation
- Debugging explanations
- Tool-call generation
- General coherence

Benchmark prompts are not included in the training corpus.

## Limitations

Scale-only training cannot change the frozen low-bit code assigned to each weight. More data can find better scale compromises, but it cannot recover all information lost during 2-bit quantization.

Closing the remaining gap to a stronger W4A8 model may require:

- Larger balanced behavioral corpora
- Teacher-KL replay from W4A8
- Direct language-model replay to protect perplexity
- Persisted RMSNorm and bias tuning
- Smaller scale groups such as G16
- Blockwise discrete-code correction
- Optional low-rank residual adapters

## Privacy and Artifact Policy

This repository must not contain:

- Hugging Face or GitHub tokens
- Private datasets
- Teacher caches
- Model checkpoints
- Safetensors shards
- Training logs containing private examples

Only source code, configuration examples, synthetic fixtures, and aggregate benchmark results should be committed.

## Reproducibility Checklist

- Record the source artifact hash.
- Record quantization format, group size, and bit width.
- Keep training and validation examples disjoint.
- Verify gradients reach `weight_scale` before training.
- Evaluate merged artifacts, not only in-memory training models.
- Preserve baseline, best-PPL, best-behavior, and final checkpoints.
- Run coherence and serving-runtime tests before publishing an artifact.

## Project Direction

The next major experiment is a larger mixed-objective run that uses W4A8 as a behavioral teacher over the Pi Office curriculum. The goal is to preserve the size and packed format of W2 while approaching the reasoning, coding, tool-use, and perplexity performance of W4A8.
