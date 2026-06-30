# FP8-K/NVFP4-V XQA Cubin

`fp8_k_nvfp4_v_h128_g4_p16.cubin` is a TensorRT-LLM XQA specialization for
`sm_100a`, BF16 query/output, head dimension 128, four query heads per KV head,
and page size 16. K is FP8 E4M3; V is packed NVFP4 E2M1 with one E4M3 scale per
16 values.

The cubin is built from `cpp/kernels/xqa_fp8k_nvfp4v` in the corresponding
TensorRT-LLM feature branch. Its SHA256 is:

```text
1250d87fa1268a44730f3ef80c8996e1b677a3d8222582b46951cddb3ca42e70
```

FlashInfer selects it only for the exact specialization above. Other mixed-KV
shapes continue to compile the source XQA implementation. The host launcher
loads this file through FlashInfer's checksum-verified cubin callback, including
when network downloads are disabled.
