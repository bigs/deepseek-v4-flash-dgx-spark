# Expert Read Benchmark

Generated: `2026-05-01T17:48:48.748977+00:00`

## Setup

- Model dir: `/home/cole/models/deepseek-v4-flash/hf`
- Tensor CSV: `/home/cole/work/deepseek-v4-flash-dgx-spark/results/spark-66c9/checkpoint-tensors.csv`
- Layers: `[0, 18, 42]`
- Methods: `['pread', 'mmap']`
- Chunk size: `8.00 MiB`
- Repeat: `2`

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
| layer0_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0252 | 2.966 | 0 | 12884 |
| layer0_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0182 | 4.230 | 0 | 7008 |
| layer0_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0249 | 2.996 | 0 | 12884 |
| layer0_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0212 | 3.520 | 0 | 10432 |
| layer0_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0258 | 2.892 | 0 | 12884 |
| layer0_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0211 | 3.537 | 0 | 10432 |
| layer0_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0258 | 2.899 | 0 | 12884 |
| layer0_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0213 | 3.515 | 0 | 10432 |
| layer0_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0206 | 3.629 | 0 | 10980 |
| layer0_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0242 | 3.339 | 0 | 12992 |
| layer0_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0204 | 3.666 | 0 | 10981 |
| layer0_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0178 | 4.200 | 0 | 8512 |
| layer0_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0252 | 2.965 | 0 | 12888 |
| layer0_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0213 | 3.503 | 0 | 10432 |
| layer0_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0259 | 2.917 | 0 | 13294 |
| layer0_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0219 | 3.444 | 0 | 10816 |
| layer0_spread_6_gap0 | mmap | 76.50 MiB | 0.0136 | 5.499 | 0 | 2472 |
| layer0_spread_6_gap0 | pread | 76.50 MiB | 0.0130 | 6.241 | 0 | 3424 |
| layer0_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0136 | 5.500 | 0 | 2472 |
| layer0_spread_6_gap1048576 | pread | 76.50 MiB | 0.0095 | 7.841 | 0 | 0 |
| layer0_whole_layer_gap0 | mmap | 3.19 GiB | 0.4471 | 7.130 | 0 | 112580 |
| layer0_whole_layer_gap0 | pread | 3.19 GiB | 0.3090 | 10.314 | 0 | 8128 |
| layer18_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0248 | 3.008 | 0 | 12884 |
| layer18_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0176 | 4.368 | 0 | 7008 |
| layer18_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0245 | 3.050 | 0 | 12884 |
| layer18_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0209 | 3.581 | 0 | 10432 |
| layer18_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0272 | 2.749 | 0 | 12884 |
| layer18_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0224 | 3.340 | 0 | 10432 |
| layer18_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0271 | 2.756 | 0 | 12884 |
| layer18_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0223 | 3.352 | 0 | 10432 |
| layer18_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0265 | 2.824 | 0 | 10980 |
| layer18_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0209 | 3.582 | 0 | 8512 |
| layer18_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0264 | 2.833 | 0 | 10980 |
| layer18_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0204 | 3.670 | 0 | 8512 |
| layer18_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0222 | 3.368 | 0 | 12888 |
| layer18_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0198 | 3.766 | 0 | 10432 |
| layer18_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0228 | 3.311 | 0 | 13295 |
| layer18_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0206 | 3.670 | 0 | 10816 |
| layer18_spread_6_gap0 | mmap | 76.50 MiB | 0.0097 | 7.671 | 0 | 2472 |
| layer18_spread_6_gap0 | pread | 76.50 MiB | 0.0106 | 7.913 | 0 | 3424 |
| layer18_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0097 | 7.693 | 0 | 2472 |
| layer18_spread_6_gap1048576 | pread | 76.50 MiB | 0.0072 | 10.344 | 0 | 0 |
| layer18_whole_layer_gap0 | mmap | 3.19 GiB | 0.3858 | 8.262 | 0 | 112580 |
| layer18_whole_layer_gap0 | pread | 3.19 GiB | 0.2761 | 11.543 | 0 | 8128 |
| layer42_file_adjacent_first_6_gap0 | mmap | 76.50 MiB | 0.0186 | 4.007 | 0 | 10576 |
| layer42_file_adjacent_first_6_gap0 | pread | 76.50 MiB | 0.0161 | 4.842 | 0 | 7008 |
| layer42_file_adjacent_first_6_gap1048576 | mmap | 76.50 MiB | 0.0187 | 3.991 | 0 | 10576 |
| layer42_file_adjacent_first_6_gap1048576 | pread | 76.50 MiB | 0.0195 | 3.841 | 0 | 10432 |
| layer42_file_adjacent_mid_6_gap0 | mmap | 76.50 MiB | 0.0187 | 3.990 | 0 | 10574 |
| layer42_file_adjacent_mid_6_gap0 | pread | 76.50 MiB | 0.0193 | 3.879 | 0 | 10432 |
| layer42_file_adjacent_mid_6_gap1048576 | mmap | 76.50 MiB | 0.0184 | 4.052 | 0 | 10574 |
| layer42_file_adjacent_mid_6_gap1048576 | pread | 76.50 MiB | 0.0194 | 3.859 | 0 | 10432 |
| layer42_numeric_0_5_gap0 | mmap | 76.50 MiB | 0.0159 | 4.711 | 0 | 8610 |
| layer42_numeric_0_5_gap0 | pread | 76.50 MiB | 0.0167 | 4.478 | 0 | 8512 |
| layer42_numeric_0_5_gap1048576 | mmap | 76.50 MiB | 0.0158 | 4.729 | 0 | 8610 |
| layer42_numeric_0_5_gap1048576 | pread | 76.50 MiB | 0.0166 | 4.493 | 0 | 8512 |
| layer42_numeric_mid_6_gap0 | mmap | 76.50 MiB | 0.0189 | 3.951 | 0 | 10576 |
| layer42_numeric_mid_6_gap0 | pread | 76.50 MiB | 0.0193 | 3.864 | 0 | 10432 |
| layer42_numeric_mid_6_gap1048576 | mmap | 77.25 MiB | 0.0194 | 3.892 | 0 | 10960 |
| layer42_numeric_mid_6_gap1048576 | pread | 77.25 MiB | 0.0199 | 3.785 | 0 | 10816 |
| layer42_spread_6_gap0 | mmap | 76.50 MiB | 0.0048 | 15.716 | 0 | 166 |
| layer42_spread_6_gap0 | pread | 76.50 MiB | 0.0089 | 9.780 | 0 | 3424 |
| layer42_spread_6_gap1048576 | mmap | 76.50 MiB | 0.0047 | 15.787 | 0 | 166 |
| layer42_spread_6_gap1048576 | pread | 76.50 MiB | 0.0055 | 13.622 | 0 | 0 |
| layer42_whole_layer_gap0 | mmap | 3.19 GiB | 0.2097 | 15.199 | 0 | 13310 |
| layer42_whole_layer_gap0 | pread | 3.19 GiB | 0.2473 | 12.889 | 0 | 8128 |
