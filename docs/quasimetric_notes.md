logits_per_image = logit_scale * image_features @ text_features.T
logits_per_text = logit_scale * text_features @ image_features.T

becomes ...

# Create IQE
d_iqe = torchqmet.IQE(
    input_size=128,            # 128-d latent vectors
    dim_per_component=16,      # split 128 dimensions into 16-dimenional chunks, where each chunk
                               #    gives an IQE component (IQE paper recommends `dim_per_component >= 8`)
).to(device)

# latents, usually from an encoder. use random vectors as example
x = torch.randn(2, 128, device=device)
y = torch.randn(2, 128, device=device)

# distance
print(d_iqe(x, y))
# tensor([22.5079, 29.4083], device='cuda:0')

# cdist
print(d_iqe(x[:, None], y))
# tensor([[22.5079, 25.2227],
#         [28.1123, 29.4083]], device='cuda:0')

# pdist
print(d_iqe(x[:, None], x))
# tensor([[ 0.0000, 22.9859],
#         [29.4122,  0.0000]], device='cuda:0')


torch.Size([128, 128]) torch.Size([128, 128])
dtype = torch.float16