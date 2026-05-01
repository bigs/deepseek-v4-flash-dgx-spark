# Expert Read Benchmark

Generated: `2026-05-01T17:50:16.536613+00:00`

## Setup

- Model dir: `/home/cole/models/deepseek-v4-flash/hf`
- Tensor CSV: `/home/cole/work/deepseek-v4-flash-dgx-spark/results/spark-66c9/checkpoint-tensors.csv`
- Layers: `[0, 18, 42]`
- Methods: `['pread', 'mmap']`
- Chunk size: `8.00 MiB`
- Repeat: `1`
- Evict mode: `iteration`

## Planned Scenarios

| scenario | kind | experts | tensor ranges | read ranges | useful | planned read | amplification |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| layer0_numeric_0_5_gap0 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_numeric_0_5_gap1048576 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_numeric_mid_6_gap0 | topk_experts | 125,126,127,128,129,130 | 36 | 4 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_numeric_mid_6_gap1048576 | topk_experts | 125,126,127,128,129,130 | 36 | 3 | 76.50 MiB | 77.25 MiB | 1.01 |
| layer0_spread_6_gap0 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_spread_6_gap1048576 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_file_adjacent_first_6_gap0 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_file_adjacent_first_6_gap1048576 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_file_adjacent_mid_6_gap0 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_file_adjacent_mid_6_gap1048576 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer0_whole_layer_gap0 | whole_layer | all | 1536 | 2 | 3.19 GiB | 3.19 GiB | 1.00 |
| layer18_numeric_0_5_gap0 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_numeric_0_5_gap1048576 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_numeric_mid_6_gap0 | topk_experts | 125,126,127,128,129,130 | 36 | 4 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_numeric_mid_6_gap1048576 | topk_experts | 125,126,127,128,129,130 | 36 | 3 | 76.50 MiB | 77.25 MiB | 1.01 |
| layer18_spread_6_gap0 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_spread_6_gap1048576 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_file_adjacent_first_6_gap0 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_file_adjacent_first_6_gap1048576 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_file_adjacent_mid_6_gap0 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_file_adjacent_mid_6_gap1048576 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer18_whole_layer_gap0 | whole_layer | all | 1536 | 2 | 3.19 GiB | 3.19 GiB | 1.00 |
| layer42_numeric_0_5_gap0 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_numeric_0_5_gap1048576 | topk_experts | 0,1,2,3,4,5 | 36 | 10 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_numeric_mid_6_gap0 | topk_experts | 125,126,127,128,129,130 | 36 | 4 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_numeric_mid_6_gap1048576 | topk_experts | 125,126,127,128,129,130 | 36 | 3 | 76.50 MiB | 77.25 MiB | 1.01 |
| layer42_spread_6_gap0 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_spread_6_gap1048576 | topk_experts | 0,51,102,153,204,255 | 36 | 12 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_file_adjacent_first_6_gap0 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_file_adjacent_first_6_gap1048576 | topk_experts | 0,1,10,100,101,102 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_file_adjacent_mid_6_gap0 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_file_adjacent_mid_6_gap1048576 | topk_experts | 210,211,212,213,214,215 | 36 | 2 | 76.50 MiB | 76.50 MiB | 1.00 |
| layer42_whole_layer_gap0 | whole_layer | all | 1536 | 2 | 3.19 GiB | 3.19 GiB | 1.00 |

## Results

| scenario | method | read | median seconds | median GiB/s | major faults | minor faults |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| layer0_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0301 | 2.482 | 2 | 6448 |
| layer0_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0214 | 3.484 | 0 | 1792 |
| layer0_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0297 | 2.515 | 2 | 6448 |
| layer0_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0237 | 3.157 | 0 | 5216 |
| layer0_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0290 | 2.580 | 2 | 6448 |
| layer0_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0237 | 3.159 | 0 | 5216 |
| layer0_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0288 | 2.590 | 2 | 6448 |
| layer0_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0247 | 3.030 | 0 | 5216 |
| layer0_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0297 | 2.516 | 10 | 5520 |
| layer0_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0423 | 1.765 | 0 | 8731 |
| layer0_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0294 | 2.538 | 10 | 5520 |
| layer0_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0253 | 2.952 | 0 | 4256 |
| layer0_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0301 | 2.482 | 4 | 6456 |
| layer0_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0246 | 3.036 | 0 | 5216 |
| layer0_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0303 | 2.488 | 3 | 6657 |
| layer0_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0265 | 2.851 | 0 | 5408 |
| layer0_spread_6_gap0 | mmap | 76.50 MiB | 0.0221 | 3.387 | 12 | 1272 |
| layer0_spread_6_gap0 | pread | 76.50 MiB | 0.0283 | 2.641 | 0 | 3424 |
| layer0_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0219 | 3.409 | 12 | 1272 |
| layer0_spread_6_gap1048576 | pread | 76.50 MiB | 0.0181 | 4.126 | 0 | 0 |
| layer0_whole_layer_gap0 | mmap | 3.19 GiB | 0.6273 | 5.081 | 2 | 56296 |
| layer0_whole_layer_gap0 | pread | 3.19 GiB | 0.5561 | 5.732 | 0 | 4064 |
| layer18_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0274 | 2.731 | 2 | 6447 |
| layer18_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0180 | 4.160 | 0 | 1792 |
| layer18_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0273 | 2.734 | 2 | 6447 |
| layer18_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0220 | 3.402 | 0 | 5216 |
| layer18_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0275 | 2.719 | 2 | 6447 |
| layer18_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0234 | 3.196 | 0 | 5217 |
| layer18_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0265 | 2.818 | 2 | 6447 |
| layer18_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0234 | 3.190 | 0 | 5216 |
| layer18_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0284 | 2.632 | 10 | 5515 |
| layer18_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0278 | 2.688 | 0 | 4257 |
| layer18_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0283 | 2.636 | 10 | 5515 |
| layer18_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0246 | 3.034 | 0 | 4256 |
| layer18_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0283 | 2.638 | 4 | 6455 |
| layer18_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0237 | 3.155 | 0 | 5216 |
| layer18_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0289 | 2.609 | 3 | 6654 |
| layer18_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0266 | 2.833 | 0 | 5409 |
| layer18_spread_6_gap0 | mmap | 76.50 MiB | 0.0181 | 4.125 | 12 | 1266 |
| layer18_spread_6_gap0 | pread | 76.50 MiB | 0.0232 | 3.224 | 0 | 3424 |
| layer18_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0180 | 4.148 | 12 | 1268 |
| layer18_spread_6_gap1048576 | pread | 76.50 MiB | 0.0156 | 4.777 | 0 | 0 |
| layer18_whole_layer_gap0 | mmap | 3.19 GiB | 0.5904 | 5.399 | 2 | 56296 |
| layer18_whole_layer_gap0 | pread | 3.19 GiB | 0.3945 | 8.080 | 0 | 4064 |
| layer42_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0274 | 2.727 | 2 | 6404 |
| layer42_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0183 | 4.079 | 0 | 1792 |
| layer42_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0266 | 2.805 | 2 | 6405 |
| layer42_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0231 | 3.239 | 0 | 5216 |
| layer42_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0261 | 2.861 | 2 | 6372 |
| layer42_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0232 | 3.221 | 0 | 5216 |
| layer42_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0268 | 2.783 | 2 | 6372 |
| layer42_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0224 | 3.329 | 0 | 5216 |
| layer42_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0255 | 2.932 | 5 | 5266 |
| layer42_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0240 | 3.111 | 0 | 4256 |
| layer42_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0243 | 3.070 | 5 | 5266 |
| layer42_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0211 | 3.538 | 0 | 4256 |
| layer42_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0291 | 2.563 | 2 | 6311 |
| layer42_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0222 | 3.367 | 0 | 5217 |
| layer42_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0263 | 2.871 | 3 | 6534 |
| layer42_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0230 | 3.283 | 0 | 5408 |
| layer42_spread_6_gap0 | mmap | 76.50 MiB | 0.0145 | 5.135 | 6 | 1009 |
| layer42_spread_6_gap0 | pread | 76.50 MiB | 0.0180 | 4.152 | 0 | 3424 |
| layer42_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0144 | 5.194 | 6 | 1009 |
| layer42_spread_6_gap1048576 | pread | 76.50 MiB | 0.0126 | 5.924 | 0 | 1 |
| layer42_whole_layer_gap0 | mmap | 3.19 GiB | 0.5980 | 5.330 | 2 | 56255 |
| layer42_whole_layer_gap0 | pread | 3.19 GiB | 0.4042 | 7.887 | 0 | 4065 |
