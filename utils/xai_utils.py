import torch
import numpy as np

def reshape_gwn_output(output):
    output=output.squeeze(dim=0)
    return torch.movedim(output, (0,1,2),(2,0,1))

def integrated_gradients(x, model, n_steps, baseline_x, temporal_focus=None, spatial_focus=None, gwn=False):
    x.requires_grad = True
    x_diff = x - baseline_x

    for k in range(1, n_steps):
        numerator_scale = k / n_steps
        curr_x = baseline_x + numerator_scale * x_diff
        y = model(curr_x)
        if gwn:
            y = reshape_gwn_output(y)
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

    if gwn:
        integrated_gradients = x_diff * integrated_gradients[0]
        integrated_gradients=reshape_gwn_output(integrated_gradients).detach().numpy()
    else:
        integrated_gradients = x_diff.detach().numpy() * integrated_gradients[0].detach().numpy()
    return (integrated_gradients)
