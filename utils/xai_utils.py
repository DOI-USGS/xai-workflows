import torch
import numpy as np

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

def expected_gradients(x, x_set, adj_matrix, model, n_samples, temporal_focus=None, spatial_focus=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_series = x_set.shape[0]
    n_segs = adj_matrix.shape[0]
    num_vars = x_set.shape[2]
    seq_len = x_set.shape[1]

    x_set_4D = x_set.reshape(n_series//n_segs,n_segs,seq_len,num_vars)

    for k in range(n_samples):
        # SAMPLE A RANDOM BASELINE INPUT
        baseline_x = torch.empty(n_segs, seq_len, num_vars).to(device)
        # for all segments
        for seg in range(n_segs):
            # pick one of the years
            seg_year = np.random.choice(n_series//n_segs)  #do we want to be sampling random segs or random years?
            # fill the baseline with a random year of data from each segment
            baseline_x[seg] = x_set_4D[seg_year, seg]

        # SAMPLE A RANDOM SCALE ALONG THE DIFFERENCE
        scale = np.random.uniform()

        # SAME IG CALCULATION
        x_diff = x - baseline_x
        curr_x = baseline_x + scale*x_diff
        if curr_x.requires_grad == False:
            curr_x.requires_grad = True
        model.zero_grad()
        y = model(curr_x)

        # GET GRADIENT
        if temporal_focus == None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, :, :], curr_x, torch.ones_like(y[:, :, :]))
        elif temporal_focus == None and spatial_focus != None:
            gradients = torch.autograd.grad(y[spatial_focus, :, :], curr_x, torch.ones_like(y[spatial_focus, :, :]))
        elif temporal_focus != None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, temporal_focus, :], curr_x, torch.ones_like(y[:, temporal_focus, :]))
        else:
            gradients = torch.autograd.grad(y[spatial_focus, temporal_focus, :], curr_x, torch.ones_like(y[spatial_focus, temporal_focus, :]))

        if k == 0:
            expected_gradients = x_diff*gradients[0] * 1/n_samples
        else:
            expected_gradients = expected_gradients + ((x_diff*gradients[0]) * 1/n_samples)

    return(expected_gradients.detach().cpu().numpy())
