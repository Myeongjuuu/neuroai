# Llama 3.2 EEG integration (initial scope)

`llama32-integration` adds `-m llama32` for the **default EEG
`mental_arithmetic` dataset**. The Zyma source labels sort as
`mental_arithmetic=0` and `rest=1`; the prompt offers A and B in that order.
The model projects 16 pooled EEG time bins into Llama's embedding space,
prepends those embeddings to a chat prompt, and returns Llama's next-token
logits for A and B. NeuralBench trains the EEG projector with its existing
cross-entropy loss and evaluates with the unchanged subject split and metrics.
The Llama backbone is frozen. This is a new integration baseline, not a BLPM
model or a reproduction of its training recipe.

Install the repository's NeuralBench/NeuralTrain packages and the optional
Llama dependency in a Python 3.12 environment. Access to
`meta-llama/Llama-3.2-3B-Instruct` on Hugging Face is required.

```bash
cd neuralbench-repo
pip install -e '../neuraltrain-repo' -e '.[llama]'
huggingface-cli login
neuralbench eeg mental_arithmetic -m llama32 --debug
```

The model config is `neuralbench/models/llama32.yaml`. Its default uses BF16
and batch size 2; change the model ID or dtype there if needed. The first
debug run downloads the checkpoint and prepares the dataset, so it is not
lightweight. NeuralBench checkpoints omit the frozen Llama weights and must
be loaded with access to the same Hugging Face model. Other tasks and dataset
variants are rejected until their class-to-answer mapping is specified and
checked.
