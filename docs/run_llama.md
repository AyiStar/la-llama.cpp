# Build and run llama.cpp on x86 CPU

## Build
Compile llama.cpp with GNU make:
```bash
# in the root dir of llama.cpp
make -j16
```
The default target will build a `main` executable
```
./main -h

usage: ./main [options]

options:
  -h, --help            show this help message and exit
  --version             show version and build info
  -i, --interactive     run in interactive mode
  --interactive-first   run in interactive mode and wait for input right away
  -ins, --instruct      run in instruction mode (use with Alpaca models)
  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)
  --multiline-input     allows you to write or paste multiple lines without ending each in '\'
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
  --color               colorise output to distinguish prompt and user input from generations
  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
  -t N, --threads N     number of threads to use during generation (default: 28)
  -tb N, --threads-batch N
                        number of threads to use during batch and prompt processing (default: same as --threads)
  -td N, --threads-draft N                        number of threads to use during generation (default: same as --threads)
  -tbd N, --threads-batch-draft N
                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: empty)
  -e, --escape          process prompt escapes sequences (\n, \r, \t, \', \", \\)
  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
  --prompt-cache-all    if specified, saves user input and generations to cache as well.
                        not supported with --interactive or other interactive options
  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.
  --random-prompt       start with a randomized prompt.
  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string
  --in-prefix STRING    string to prefix user inputs with (default: empty)
  --in-suffix STRING    string to suffix after user inputs with (default: empty)
  -f FNAME, --file FNAME
                        prompt file to start generation.
  -bf FNAME, --binary-file FNAME
                        binary file containing multiple choice tasks.
  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  -c N, --ctx-size N    size of the prompt context (default: 512, 0 = loaded from model)
  -b N, --batch-size N  logical maximum batch size (default: 2048)
  -ub N, --ubatch-size N
                        physical maximum batch size (default: 512)
  --samplers            samplers that will be used for generation in the order, separated by ';'
                        (default: top_k;tfs_z;typical_p;top_p;min_p;temperature)
  --sampling-seq        simplified sequence for samplers that will be used (default: kfypmt)
  --top-k N             top-k sampling (default: 40, 0 = disabled)
  --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
  --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
  --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)
  --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
  --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
  --dynatemp-range N    dynamic temperature range (default: 0.0, 0.0 = disabled)
  --dynatemp-exp N      dynamic temperature exponent (default: 1.0)
  --mirostat N          use Mirostat sampling.
                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
  --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
  --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
                        modifies the likelihood of token appearing in the completion,
                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)
  --grammar-file FNAME  file to read grammar from
  --cfg-negative-prompt PROMPT
                        negative prompt to use for guidance. (default: empty)
  --cfg-negative-prompt-file FNAME
                        negative prompt file to use for guidance. (default: empty)
  --cfg-scale N         strength of guidance (default: 1.000000, 1.0 = disable)
  --rope-scaling {none,linear,yarn}
                        RoPE frequency scaling method, defaults to linear unless specified by the model
  --rope-scale N        RoPE context scaling factor, expands context by a factor of N
  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N
  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)
  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: 1.0)
  --yarn-beta-fast N    YaRN: low correction dim or beta (default: 32.0)
  --pooling {none,mean,cls}
                        pooling type for embeddings, use model default if unspecified
  -dt N, --defrag-thold N
                        KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
  --no-penalize-nl      do not penalize newline token
  --temp N              temperature (default: 0.8)
  --all-logits          return logits for all tokens in the batch (default: disabled)
  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f
  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: 400)
  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f
  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: 0)
  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f
  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: 0)
  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base
  --keep N              number of tokens to keep from the initial prompt (default: 0, -1 = all)
  --draft N             number of tokens to draft for speculative decoding (default: 5)
  --chunks N            max number of chunks to process (default: -1, -1 = all)
  -np N, --parallel N   number of parallel sequences to decode (default: 1)
  -ns N, --sequences N  number of sequences to decode (default: 1)
  -ps N, --p-split N    speculative decoding split probability (default: 0.1)
  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)
  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md
  --image IMAGE_FILE    path to an image file. use with multimodal models
  --mlock               force system to keep model in RAM rather than swapping or compressing
  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
  --numa TYPE           attempt optimizations that help on some NUMA systems
                          - distribute: spread execution evenly over all nodes
                          - isolate: only spawn threads on CPUs on the node that execution started on
                          - numactl: use the CPU map provided by numactl
                        if run without this previously, it is recommended to drop the system page cache before using this
                        see https://github.com/ggerganov/llama.cpp/issues/1437
  --verbose-prompt      print a verbose prompt before generation (default: false)
  --no-display-prompt   don't print prompt at generation (default: false)
  -gan N, --grp-attn-n N
                        group-attention factor (default: 1)
  -gaw N, --grp-attn-w N
                        group-attention width (default: 512.0)
  -dkvc, --dump-kv-cache
                        verbose print of the KV cache
  -nkvo, --no-kv-offload
                        disable KV offload
  -ctk TYPE, --cache-type-k TYPE
                        KV cache data type for K (default: f16)
  -ctv TYPE, --cache-type-v TYPE
                        KV cache data type for V (default: f16)
  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles
  --lora FNAME          apply LoRA adapter (implies --no-mmap)
  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)
  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
  -m FNAME, --model FNAME
                        model path (default: models/7B/ggml-model-f16.gguf)
  -md FNAME, --model-draft FNAME
                        draft model for speculative decoding
  -ld LOGDIR, --logdir LOGDIR
                        path under which to save YAML logs (no logging if unset)
  --override-kv KEY=TYPE:VALUE
                        advanced option to override model metadata by key. may be specified multiple times.
                        types: int, float, bool. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
  -ptc N, --print-token-count N
                        print token count every N tokens (default: -1)

log options:
  --log-test            Run simple logging test
  --log-disable         Disable trace logs
  --log-enable          Enable trace logs
  --log-file            Specify a log filename (without extension)
  --log-new             Create a separate new log file on start. Each log file will have unique name: "<name>.<ID>.log"
  --log-append          Don't truncate the old log file.
```

## Model Weights
Llama.cpp works with GGUF format (originally GGML format). We can either
- download original llama2 weights and convert them to GGUF with provided python scripts, or
- directly download converted GGUF-format files.
Both can be downloaded with huggingface hub, see `scripts/llama_weights_download.py`.
You can opitonally use a mirror by setting `HF_ENDPOINT` to `https://hf-mirror.com`.

```bash
export LLAMA_GGUF_PATH = <path_to_downloaded_gguf_model>

tree $LLAMA_GGUF_PATH/Llama-2-7B-GGUF/
|-- LICENSE.txt
|-- Notice
|-- README.md
|-- USE_POLICY.md
|-- config.json
|-- llama-2-7b.Q2_K.gguf
|-- llama-2-7b.Q3_K_L.gguf
|-- llama-2-7b.Q3_K_M.gguf
|-- llama-2-7b.Q3_K_S.gguf
|-- llama-2-7b.Q4_0.gguf
|-- llama-2-7b.Q4_K_M.gguf
|-- llama-2-7b.Q4_K_S.gguf
|-- llama-2-7b.Q5_0.gguf
|-- llama-2-7b.Q5_K_M.gguf
|-- llama-2-7b.Q5_K_S.gguf
|-- llama-2-7b.Q6_K.gguf
`-- llama-2-7b.Q8_0.gguf
0 directories, 17 files
```

## Run a model
Run llama-2-7b.Q4_0.gguf as an example.
```bash
./main -m $LLAMA_GGUF_PATH/llama-2-7b.Q4_0.gguf -n 512 -p "Building a website can be done in 10 simple steps:\nStep 1:" -e -t 1
```
