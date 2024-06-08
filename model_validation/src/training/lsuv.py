import torch

def LSUV_(model, data, apply_only_to=['Conv', 'Linear', 'Bilinear'],
          std_tol=0.1, max_iters=10, do_ortho_init=True, logging_FN=print):
    r"""
    Applies layer sequential unit variance (LSUV), as described in
    `All you need is a good init` - Mishkin, D. et al (2015):
    https://arxiv.org/abs/1511.06422

    Args:
        model: `torch.nn.Module` object on which to apply LSUV.
        data: sample input data drawn from training dataset.
        apply_only_to: list of strings indicating target children
            modules. For example, ['Conv'] results in LSUV applied
            to children of type containing the substring 'Conv'.
        std_tol: positive number < 1.0, below which differences between
            actual and unit standard deviation are acceptable.
        max_iters: number of times to try scaling standard deviation
            of each children module's output activations.
        do_ortho_init: boolean indicating whether to apply orthogonal
            init to parameters of dim >= 2 (zero init if dim < 2).
        logging_FN: function for outputting progress information.

    Example:
        >>> model = nn.Sequential(nn.Linear(8, 2), nn.Softmax(dim=1))
        >>> data = torch.randn(100, 8)
        >>> LSUV_(model, data)
    """

    matched_modules = [m for m in model.modules() if any(substr in str(type(m)) for substr in apply_only_to)]

    if do_ortho_init:
        logging_FN(f"Applying orthogonal init (zero init if dim < 2) to params in {len(matched_modules)} module(s).")
        for m in matched_modules:
            for p in m.parameters():                
                torch.nn.init.orthogonal_(p) if (p.dim() >= 2) else torch.nn.init.zeros_(p)

    logging_FN(f"Applying LSUV to {len(matched_modules)} module(s) (up to {max_iters} iters per module):")

    def _compute_and_store_LSUV_stats(m, inp, out):
        m._LSUV_stats = { 'mean': out.detach().mean(), 'std': out.detach().std() }

    was_training = model.training
    model.train()  # sets all modules to training behavior
    with torch.no_grad():
        for i, m in enumerate(matched_modules):
            with m.register_forward_hook(_compute_and_store_LSUV_stats):
                for t in range(max_iters):
                    _ = model(data)  # run data through model to get stats
                    mean, std = m._LSUV_stats['mean'], m._LSUV_stats['std']
                    if abs(std - 1.0) < std_tol:
                        break
                    m.weight.data /= (std + 1e-6)
            logging_FN(f"Module {i:2} after {(t+1):2} itr(s) | Mean:{mean:7.3f} | Std:{std:6.3f} | {type(m)}")
            delattr(m, '_LSUV_stats')

    if not was_training: model.eval()