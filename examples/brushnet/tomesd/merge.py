import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


import torch
import numpy as np
import cv2

def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
                metric.device)
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], int(N * r))

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2,
                     index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge

def bipartite_soft_matching_mask_random2d(metric: torch.Tensor,
                                               w: int, h: int, sx: int, sy: int, r1: float, r2: float,
                                               mask_image: torch.Tensor,
                                               no_rand: bool = False,
                                               generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Extends bipartite_soft_matching_random2d with a mask_image input to control merging ratios based on regions.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - mask_image: 64x64 mask image with 1 for white and 0 for black areas
     - no_rand: if true, disable randomness (use top left corner only)
     - generator: random generator for reproducibility
    """
    B, N, _ = metric.shape

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx
        mask_resized = torch.nn.functional.max_pool2d(mask_image.unsqueeze(0).unsqueeze(0), kernel_size=2,
                                                      stride=2).squeeze(0).squeeze(0)
        mask_resized = mask_resized.to(generator.device)
        # Adjust ratio for each region based on the mask
        region_ratios = torch.where(mask_resized > 0, 0.4, 0.9)
        # Calculate r dynamically based on ratios
        num_tokens = metric.shape[1]
        ratio_mean = region_ratios.mean().item()  # Average ratio across all regions
        r = int(num_tokens * ratio_mean)  # Compute r using the average ratio

        if r <= 0:
            return do_nothing, do_nothing

        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
                metric.device)
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        #import pdb; pdb.set_trace()
        mask_idx_buffer_view=idx_buffer_view * mask_resized.unsqueeze(dim=2) + 1- mask_resized.unsqueeze(dim=2).to(torch.int64)
        non_mask_idx_buffer_view=idx_buffer_view *(1-mask_resized.unsqueeze(dim=2)) + mask_resized.unsqueeze(dim=2).to(torch.int64)

        mask_idx_buffer_view = mask_idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)
        non_mask_idx_buffer_view = non_mask_idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            mask_idx_buffer = mask_idx_buffer_view
            non_mask_idx_buffer = non_mask_idx_buffer_view

        mask_rand_idx = mask_idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        non_mask_rand_idx = non_mask_idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        del mask_idx_buffer_view, non_mask_idx_buffer_view

        if (non_mask_idx_buffer==-1).sum().item() + (mask_idx_buffer==-1).sum().item() == hsy*wsx:
            mask_num_dst = (mask_idx_buffer==-1).sum().item()
            non_mask_num_dst = (non_mask_idx_buffer==-1).sum().item()
            non_mask_num = (mask_idx_buffer==1).sum().item()
            mask_num = (non_mask_idx_buffer==1).sum().item()

        del mask_idx_buffer, non_mask_idx_buffer
        mask_a_idx = mask_rand_idx[:, mask_num_dst:-non_mask_num, :] # src idx
        mask_b_idx = mask_rand_idx[:, :mask_num_dst, :] # dst dx

        non_mask_a_idx = non_mask_rand_idx[:, non_mask_num_dst:-mask_num, :]
        non_mask_b_idx = non_mask_rand_idx[:, :non_mask_num_dst, :]

        del mask_rand_idx, non_mask_rand_idx
        # -------------- 11/21 구현 구역 ----------

        def split(x, a_idx, b_idx, num_dst, num_src):
            """
            Splits the input tensor x into src and dst based on provided indices.
            Args:
                x: Input tensor [B, N, C].
                a_idx: Indices for source tokens.
                b_idx: Indices for destination tokens.
                num_dst: Number of destination tokens.
                num_src: Number of source tokens.
            Returns:
                src: Source tokens.
                dst: Destination tokens.
            """
            B, N, C = x.shape

            # Validate input dimensions
            if num_src + num_dst > N:
                raise ValueError(f"num_src ({num_src}) + num_dst ({num_dst}) exceeds input size ({N}).")
            # Adjust a_idx and b_idx if necessary
            if a_idx.shape[1] != num_src:
                a_idx = a_idx[:, :num_src, :]
            if b_idx.shape[1] != num_dst:
                b_idx = b_idx[:, :num_dst, :]

            # Expand indices to match x
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        mask_a, mask_b = split(metric, mask_a_idx, mask_b_idx, mask_num_dst, N - mask_num_dst - non_mask_num)
        non_mask_a, non_mask_b = split(metric, non_mask_a_idx, non_mask_b_idx, non_mask_num_dst, N - non_mask_num_dst - mask_num)

        mask_scores = mask_a @ mask_b.transpose(-1, -2)
        non_mask_scores = non_mask_a @ non_mask_b.transpose(-1, -2)
        mask_r = min(mask_a.shape[1], int((N-non_mask_num) * r1)) #mask_r
        non_mask_r = min(non_mask_a.shape[1], int((N-mask_num) * r2)) #non_mask_r

        mask_node_max, mask_node_idx = mask_scores.max(dim=-1)
        non_mask_node_max, non_mask_node_idx = non_mask_scores.max(dim=-1)
        mask_edge_idx = mask_node_max.argsort(dim=-1, descending=True)[..., None]
        non_mask_edge_idx = non_mask_node_max.argsort(dim=-1, descending=True)[..., None]

        mask_unm_idx = mask_edge_idx[..., mask_r:, :]
        non_mask_unm_idx = non_mask_edge_idx[..., non_mask_r:, :]
        mask_src_idx = mask_edge_idx[..., :mask_r, :]
        non_mask_src_idx = non_mask_edge_idx[..., :non_mask_r, :]
        mask_dst_idx = gather(mask_node_idx[..., None], dim=-2, index=mask_src_idx)
        non_mask_dst_idx = gather(non_mask_node_idx[..., None], dim=-2, index=non_mask_src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        mask_src, mask_dst = split(x, mask_a_idx, mask_b_idx, mask_num_dst, N - mask_num_dst - non_mask_num)
        non_mask_src, non_mask_dst = split(x, non_mask_a_idx, non_mask_b_idx, non_mask_num_dst, N - non_mask_num_dst - mask_num)
        n1, t1, c1 = mask_src.shape
        n2, t2, c2 = non_mask_src.shape
        mask_unm = gather(mask_src, dim=-2, index=mask_unm_idx.expand(n1, t1 - mask_r, c1))
        non_mask_unm = gather(non_mask_src, dim=-2, index=non_mask_unm_idx.expand(n2, t2 - non_mask_r, c2))
        unm = torch.cat([mask_unm, non_mask_unm], dim=1)
        del mask_unm, non_mask_unm

        mask_src = gather(mask_src, dim=-2, index=mask_src_idx.expand(n1, mask_r, c1))
        non_mask_src = gather(non_mask_src, dim=-2, index=non_mask_src_idx.expand(n2, non_mask_r, c2))

        mask_dst = mask_dst.scatter_reduce(-2, mask_dst_idx.expand(n1, mask_r, c1), mask_src, reduce=mode)
        non_mask_dst = non_mask_dst.scatter_reduce(-2, non_mask_dst_idx.expand(n2, non_mask_r, c2), non_mask_src, reduce=mode)
        dst = torch.cat([mask_dst, non_mask_dst], dim=1)
        del mask_dst, non_mask_dst, mask_src, non_mask_src
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """
        Unmerge for both masked and non-masked regions.
        """

        # Define lengths based on indices
        mask_unm_len = mask_unm_idx.shape[1]
        non_mask_unm_len = non_mask_unm_idx.shape[1]

        mask_unm, non_mask_unm, mask_dst, non_mask_dst = (
            x[..., :mask_unm_len, :],
            x[..., mask_unm_len:mask_unm_len + non_mask_unm_len, :],
            x[..., mask_unm_len + non_mask_unm_len:mask_unm_len + non_mask_unm_len + mask_num_dst, :],
            x[..., mask_unm_len + non_mask_unm_len + mask_num_dst:, :]
        )

        _, _, c = mask_unm.shape

        # 복원: Masked regions
        mask_src = gather(mask_dst, dim=-2, index=mask_dst_idx.expand(B, mask_r, c))
        mask_out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        mask_out.scatter_(-2, mask_b_idx.expand(B, mask_num_dst, c), src=mask_dst)
        mask_out.scatter_(-2, gather(mask_a_idx.expand(B, mask_a_idx.shape[1], 1), dim=1, index=mask_unm_idx).expand(B,
                                                                                                                     mask_unm_len,
                                                                                                                     c),
                          src=mask_unm)
        mask_out.scatter_(-2, gather(mask_a_idx.expand(B, mask_a_idx.shape[1], 1), dim=1, index=mask_src_idx).expand(B,
                                                                                                                     mask_r,
                                                                                                                     c),
                          src=mask_src)

        # 복원: Non-masked regions
        non_mask_src = gather(non_mask_dst, dim=-2, index=non_mask_dst_idx.expand(B, non_mask_r, c))
        non_mask_out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        non_mask_out.scatter_(-2, non_mask_b_idx.expand(B, non_mask_num_dst, c), src=non_mask_dst)
        non_mask_out.scatter_(-2, gather(non_mask_a_idx.expand(B, non_mask_a_idx.shape[1], 1), dim=1,
                                         index=non_mask_unm_idx).expand(B, non_mask_unm_len, c), src=non_mask_unm)
        non_mask_out.scatter_(-2, gather(non_mask_a_idx.expand(B, non_mask_a_idx.shape[1], 1), dim=1,
                                         index=non_mask_src_idx).expand(B, non_mask_r, c), src=non_mask_src)

        # Combine both outputs
        final_output = mask_out + non_mask_out


        return final_output

    return merge, unmerge



