# Lazy Runtime Prompt Generation, 2 Tokens

Date: 2026-05-01  
Host: `spark-66c9`

Successful full-context prompt-generation probe.

Key output:

```text
max_seq_len 1048576
prompt 'Hello'
input_token_ids [19923]
generated_token_ids [21133, 438]
generated_text 'World ='
prompt_plus_generated_text 'HelloWorld ='
logits_shape (1, 129280) torch.float32
```

Memory:

```text
mem_after_load (54670131200, 130663661568)
mem_after_forward (42789363712, 130663661568)
```

Guard result:

```text
docker exited return_code=0 reason=process exited
```

This run used `deepseek-v4-flash-spark:tilelang`,
`spark_runtime.lazy_official_runtime`, and `max_seq_len=1048576`.
