# Checkpoint Inventory

Generated: `2026-05-01T17:43:03.260754+00:00`

## Model

- Model directory: `/home/cole/models/deepseek-v4-flash/hf`
- Architecture: `DeepseekV4ForCausalLM`
- Layers: `43`
- Routed experts: `256`
- Shared experts: `1`
- Experts per token: `6`
- Expert dtype: `fp4`
- Quantization: `{"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}`

## Totals

- Tensor payload: `148.65 GiB` (`159609485896` bytes)
- Shard files: `148.66 GiB` across `46` safetensors
- Tensor count: `69187`

## Bytes By Dtype

| dtype   | count | bytes        | human      |
| ------- | ----- | ------------ | ---------- |
| I8      | 33792 | 141733920768 | 132.00 GiB |
| F8_E8M0 | 34167 | 8858737664   | 8.25 GiB   |
| F8_E4M3 | 375   | 6023020544   | 5.61 GiB   |
| BF16    | 433   | 2830518528   | 2.64 GiB   |
| F32     | 417   | 144672072    | 137.97 MiB |
| I64     | 3     | 18616320     | 17.75 MiB  |

## Bytes By Component

| component             | count | bytes        | human       | largest dtypes                                      |
| --------------------- | ----- | ------------ | ----------- | --------------------------------------------------- |
| core_routed_experts   | 66048 | 147169738752 | 137.06 GiB  | I8 129.00 GiB, F8_E8M0 8.06 GiB                     |
| core_attention_dense  | 870   | 5400554112   | 5.03 GiB    | F8_E4M3 4.45 GiB, BF16 590.67 MiB, F32 5.42 MiB     |
| mtp_routed_experts    | 1536  | 3422552064   | 3.19 GiB    | I8 3.00 GiB, F8_E8M0 192.00 MiB                     |
| core_shared_experts   | 258   | 1082196480   | 1.01 GiB    | F8_E4M3 1.01 GiB, F8_E8M0 64.50 KiB                 |
| embeddings            | 1     | 1059061760   | 1010.00 MiB | BF16 1010.00 MiB                                    |
| output_head           | 1     | 1059061760   | 1010.00 MiB | BF16 1010.00 MiB                                    |
| core_hyper_correction | 258   | 135275592    | 129.01 MiB  | F32 129.01 MiB                                      |
| core_router           | 86    | 108834816    | 103.79 MiB  | BF16 86.00 MiB, I64 17.75 MiB, F32 40.00 KiB        |
| mtp_attention_dense   | 13    | 106964608    | 102.01 MiB  | F8_E4M3 102.00 MiB, F8_E8M0 6.38 KiB, BF16 3.00 KiB |
| mtp_projection        | 4     | 33556480     | 32.00 MiB   | F8_E4M3 32.00 MiB, F8_E8M0 2.00 KiB                 |
| mtp_shared_experts    | 6     | 25167360     | 24.00 MiB   | F8_E4M3 24.00 MiB, F8_E8M0 1.50 KiB                 |
| mtp_hyper_correction  | 9     | 3408108      | 3.25 MiB    | F32 3.25 MiB                                        |
| mtp_router            | 2     | 2098176      | 2.00 MiB    | BF16 2.00 MiB, F32 1.00 KiB                         |
| core_norms            | 86    | 704512       | 688.00 KiB  | BF16 688.00 KiB                                     |
| auxiliary_heads       | 3     | 262164       | 256.02 KiB  | F32 256.02 KiB                                      |
| mtp_norms             | 5     | 40960        | 40.00 KiB   | BF16 40.00 KiB                                      |
| final_norm            | 1     | 8192         | 8.00 KiB    | BF16 8.00 KiB                                       |

## Expert Matrix Bytes

| matrix                        | bytes       | human     |
| ----------------------------- | ----------- | --------- |
| core_routed_experts.w1.scale  | 2885681152  | 2.69 GiB  |
| core_routed_experts.w1.weight | 46170898432 | 43.00 GiB |
| core_routed_experts.w2.scale  | 2885681152  | 2.69 GiB  |
| core_routed_experts.w2.weight | 46170898432 | 43.00 GiB |
| core_routed_experts.w3.scale  | 2885681152  | 2.69 GiB  |
| core_routed_experts.w3.weight | 46170898432 | 43.00 GiB |
| mtp_routed_experts.w1.scale   | 67108864    | 64.00 MiB |
| mtp_routed_experts.w1.weight  | 1073741824  | 1.00 GiB  |
| mtp_routed_experts.w2.scale   | 67108864    | 64.00 MiB |
| mtp_routed_experts.w2.weight  | 1073741824  | 1.00 GiB  |
| mtp_routed_experts.w3.scale   | 67108864    | 64.00 MiB |
| mtp_routed_experts.w3.weight  | 1073741824  | 1.00 GiB  |

## Contiguity Checks

| component            | files | tensors | bytes      | runs | nonzero gaps | gap bytes  | max gap    |
| -------------------- | ----- | ------- | ---------- | ---- | ------------ | ---------- | ---------- |
| core_routed_experts  | 43    | 66048   | 137.06 GiB | 86   | 43           | 5.46 GiB   | 134.00 MiB |
| mtp_routed_experts   | 1     | 1536    | 3.19 GiB   | 2    | 1            | 158.00 MiB | 158.00 MiB |
| core_shared_experts  | 43    | 258     | 1.01 GiB   | 86   | 43           | 4.45 GiB   | 110.00 MiB |
| mtp_shared_experts   | 1     | 6       | 24.00 MiB  | 2    | 1            | 118.00 MiB | 118.00 MiB |
| core_attention_dense | 43    | 870     | 5.03 GiB   | 172  | 129          | 8.27 GiB   | 192.00 MiB |

## Core Expert Bytes By Layer

| layer | bytes      | human    |
| ----- | ---------- | -------- |
| 0     | 3422552064 | 3.19 GiB |
| 1     | 3422552064 | 3.19 GiB |
| 2     | 3422552064 | 3.19 GiB |
| 3     | 3422552064 | 3.19 GiB |
| 4     | 3422552064 | 3.19 GiB |
| 5     | 3422552064 | 3.19 GiB |
| 6     | 3422552064 | 3.19 GiB |
| 7     | 3422552064 | 3.19 GiB |
| 8     | 3422552064 | 3.19 GiB |
| 9     | 3422552064 | 3.19 GiB |
| 10    | 3422552064 | 3.19 GiB |
| 11    | 3422552064 | 3.19 GiB |
| 12    | 3422552064 | 3.19 GiB |
| 13    | 3422552064 | 3.19 GiB |
| 14    | 3422552064 | 3.19 GiB |
| 15    | 3422552064 | 3.19 GiB |
| 16    | 3422552064 | 3.19 GiB |
| 17    | 3422552064 | 3.19 GiB |
| 18    | 3422552064 | 3.19 GiB |
| 19    | 3422552064 | 3.19 GiB |
| 20    | 3422552064 | 3.19 GiB |
| 21    | 3422552064 | 3.19 GiB |
| 22    | 3422552064 | 3.19 GiB |
| 23    | 3422552064 | 3.19 GiB |
| 24    | 3422552064 | 3.19 GiB |
| 25    | 3422552064 | 3.19 GiB |
| 26    | 3422552064 | 3.19 GiB |
| 27    | 3422552064 | 3.19 GiB |
| 28    | 3422552064 | 3.19 GiB |
| 29    | 3422552064 | 3.19 GiB |
| 30    | 3422552064 | 3.19 GiB |
| 31    | 3422552064 | 3.19 GiB |
| 32    | 3422552064 | 3.19 GiB |
| 33    | 3422552064 | 3.19 GiB |
| 34    | 3422552064 | 3.19 GiB |
| 35    | 3422552064 | 3.19 GiB |
| 36    | 3422552064 | 3.19 GiB |
| 37    | 3422552064 | 3.19 GiB |
| 38    | 3422552064 | 3.19 GiB |
| 39    | 3422552064 | 3.19 GiB |
| 40    | 3422552064 | 3.19 GiB |
| 41    | 3422552064 | 3.19 GiB |
| 42    | 3422552064 | 3.19 GiB |

## Initial Interpretation

- The routed expert payload is the dominant checkpoint component and is the main target for streaming/caching.
- Expert weights are stored as `I8` payloads with `F8_E8M0` scales; this matches a packed low-bit expert representation rather than ordinary BF16 weights.
- Shared experts are small enough to keep resident with the dense weights.
- Each full core layer's routed expert payload is roughly the same size, making layer/expert-local cache accounting straightforward.
- The contiguity table shows whether component-local reads can be satisfied by long sequential ranges or require many small reads.

The companion CSV contains one row per tensor with file offsets and component classification.
