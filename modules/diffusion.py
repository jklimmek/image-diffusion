# Modules :
# - Diffusion 

# import torch


# __all__ = [
#     "Diffusion"
# ]


# class Diffusion:

#     def __init__(
#             self, 
#             encoder, 
#             decoder, 
#             codebook, 
#             scheduler, 
#         ):

#         self.encoder = encoder
#         self.decoder = decoder
#         self.codebook = codebook
#         self.scheduler = scheduler


#     @torch.no_grad()
#     def generate(dec, codebook, unet, scheduler, cfg_scales, device="cuda", seed=None):
#         if seed is not None:
#             torch.manual_seed(seed)
#         B = len(places)
#         C = len(cfg_scales)
#         cfg_scales = torch.tensor(B * cfg_scales, device=device)[:, None, None, None]
#         xt = torch.randn(B * C, z, 32, 32, device=device)
#         class_labels = torch.tensor([places[place] for place in places] * C, device=device)
#         for i in tqdm(reversed(range(1000)), total=1000, ncols=100):
#             t = torch.full((B * C,), i, dtype=torch.long, device=device)
#             noise_pred_cond = unet(xt, t, class_labels)
#             noise_pred_uncond = unet(xt, t)
#             noise_pred = noise_pred_uncond + cfg_scales * (noise_pred_cond - noise_pred_uncond)
#             xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, t)
#         z_q, _, _ = codebook(xt)
#         imgs = dec(z_q)
#         return imgs