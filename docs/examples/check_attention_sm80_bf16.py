"""Compile and check the standalone template. Requires CUDA; no reference fallback.

Run from the repository root:
  TORCH_CUDA_ARCH_LIST=8.0 python docs/examples/check_attention_sm80_bf16.py --device cuda:1
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def check(extension, q, k, v, past_len):
    out, lse = extension.forward(q, k, v, past_len)
    tq, tk = q.shape[2], k.shape[2]
    qi = torch.arange(tq, device=q.device) + past_len
    kj = torch.arange(tk, device=q.device)
    mask = kj[None, :] <= qi[:, None]
    score = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    score = score.masked_fill(~mask, -float('inf'))
    oracle = score.softmax(-1) @ v.double()
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    # P is rounded to BF16 per block before PV. This differs from FP64 and from
    # SDPA's internal rounding; report error and apply BF16-appropriate tolerances.
    torch.testing.assert_close(out.float(), oracle.float(), atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(out, sdpa, atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(lse, score.logsumexp(-1).float(), atol=2e-4, rtol=2e-5)
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert out.shape == q.shape and lse.shape == q.shape[:-1]
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    print(f"PASS Q={tuple(q.shape)} K={tuple(k.shape)} past={past_len} "
          f"max|O-FP64|={(out.double()-oracle).abs().max().item():.6g} "
          f"max|LSE-FP64|={(lse.double()-score.logsumexp(-1)).abs().max().item():.6g}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This validation requires an SM80+ CUDA GPU and nvcc')
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    include = root / 'third_party' / 'cutlass' / 'include'
    if not (include / 'cute' / 'tensor.hpp').is_file():
        raise RuntimeError('Initialize the repository CUTLASS dependency first')
    with torch.cuda.device(device):
        extension = load(
            name='attention_sm80_bf16_example',
            sources=[str(Path(__file__).with_name('attention_sm80_bf16_binding.cu'))],
            extra_include_paths=[str(include)],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                              '-lineinfo', '--ptxas-options=-v'],
            with_cuda=True, verbose=True,
        )
        torch.manual_seed(42)
        torch.set_float32_matmul_precision('highest')
        rand = lambda shape: torch.randn(shape, device=device, dtype=torch.bfloat16)
        # 1/2/3/4/5/8 KV blocks exercise prologue, alternating stages and drain,
        # along with partially filled query and key tiles.
        for tq, tk in ((1, 1), (17, 17), (64, 64), (65, 65), (129, 129),
                       (193, 193), (257, 257), (512, 512),
                       (1, 513), (7, 193), (65, 193)):
            q = rand((2, 3, tq, 64))
            # Poison the unused cache capacity to expose invalid tail reads.
            stores = [torch.full((2, 3, tk + 64, 64), float('nan'),
                                 device=device, dtype=torch.bfloat16) for _ in range(2)]
            for store in stores:
                store[:, :, :tk].copy_(rand((2, 3, tk, 64)))
            check(extension, q, stores[0][:, :, :tk], stores[1][:, :, :tk], tk - tq)

        # Contiguous views with an unaligned base and general feature strides.
        count = 2 * 3 * 65 * 64
        views = [rand((count + 1,))[1:].view(2, 3, 65, 64) for _ in range(3)]
        check(extension, *views, 0)
        views = [rand((2, 3, 65, 128))[..., ::2] for _ in range(3)]
        check(extension, *views, 0)

        # Causality and deterministic stage reuse over several iterations.
        q, k, v = [rand((2, 3, 193, 64)) for _ in range(3)]
        before = check(extension, q, k, v, 0)
        k[:, :, 65:] += 4
        v[:, :, 65:] += 4
        after = check(extension, q, k, v, 0)
        torch.testing.assert_close(before[:, :, :65], after[:, :, :65], atol=0, rtol=0)
        repeat, _ = extension.forward(q, k, v)
        torch.testing.assert_close(repeat, after, atol=0, rtol=0)

        # Exact-output case: catches missing/duplicated PV tiles and bad P layout
        # without hiding those errors behind BF16 tolerance.
        q = torch.zeros((2, 3, 193, 64), device=device, dtype=torch.bfloat16)
        k, v = torch.zeros_like(q), torch.ones_like(q)
        exact = check(extension, q, k, v, 0)
        torch.testing.assert_close(exact, torch.ones_like(exact), atol=0, rtol=0)

        # Producers, contiguous copies and async kernel all use a nondefault stream.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            q.zero_()
            k.zero_()
            v.fill_(2)
            out, _ = extension.forward(q, k, v)
            out = out.clone()
        stream.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, 2), atol=0, rtol=0)
        torch.cuda.synchronize(device)
        print('All template checks passed. Run compute-sanitizer separately for memory/race checks.')


if __name__ == '__main__':
    main()
