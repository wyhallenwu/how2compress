import time

import torch
from einops import rearrange

from src.model.model import EncodePipeline

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mit = EncodePipeline().to("cuda:1")
    mit.train()
    # for _ in range(10):
    #     image = torch.randn(1, 3, 224, 224).to("cuda:0")
    #     mit(image)
    start = time.time()
    image = torch.randn(8, 3, 32, 32).to("cuda:1")
    out = mit(image)
    end = time.time()
    print(f"Time: {end - start}")
    print(out.shape)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory: {peak_memory / 1024 ** 2} MB")
    # for o in out:
    # print(o.shape)

    # embed = OverlapPatchEmbed(
    #     img_size=(1080, 1920),
    #     patch_size=7,
    #     stride=4,
    #     in_chans=3,
    #     embed_dim=64,
    # ).to("cuda:0")
    # x = torch.randn(1, 3, 1080, 1920).to("cuda:0")
    # y, h, w = embed(x)
    # print(y.shape)
    # print(h, w)
    # y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
    # embed1 = OverlapPatchEmbed(
    #     img_size=(h, w),
    #     patch_size=7,
    #     stride=4,
    #     in_chans=64,
    #     embed_dim=128,
    # ).to("cuda:0")
    # y, h, w = embed1(y)
    # print(y.shape)
    # print(h, w)
