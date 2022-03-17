import torch

def integrated_gradients(x, model, n_steps, baseline_x, temporal_focus=None, spatial_focus=None):
    x.requires_grad = True
    x_diff = x - baseline_x

    for k in range(1, n_steps):
        numerator_scale = k / n_steps
        curr_x = baseline_x + numerator_scale * x_diff
        y = model(curr_x)

        if temporal_focus == None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, :, :], curr_x, torch.ones_like(y[:, :, :]))
        elif temporal_focus == None and spatial_focus != None:
            gradients = torch.autograd.grad(y[spatial_focus, :, :], curr_x, torch.ones_like(y[spatial_focus, :, :]))
        elif temporal_focus != None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, temporal_focus, :], curr_x, torch.ones_like(y[:, temporal_focus, :]))
        else:
            gradients = torch.autograd.grad(y[spatial_focus, temporal_focus, :], curr_x,
                                            torch.ones_like(y[spatial_focus, temporal_focus, :]))

        if k == 1:
            integrated_gradients = gradients
        else:
            integrated_gradients = integrated_gradients + gradients

    integrated_gradients = x_diff.detach().cpu().numpy() * integrated_gradients[0].detach().cpu().numpy()
    return (integrated_gradients)
