import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes and support backward propagation.
    Unlike standard torch.distributed.all_gather, this correctly routes
    gradients back to the local processes.
    """
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if not dist.is_available() or not dist.is_initialized():
            return tuple([input])
            
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        if not dist.is_available() or not dist.is_initialized():
            return grads[0]
            
        grad_out = torch.zeros_like(input)
        # Select the gradient slice corresponding to this rank
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def gather_embeddings(z):
    """
    Helper function to safely gather embeddings across all distributed processes.
    If distributed training is not active, it simply returns the input tensor.
    """
    if not dist.is_available() or not dist.is_initialized():
        return z
    
    gathered = GatherLayer.apply(z)
    return torch.cat(gathered, dim=0)
