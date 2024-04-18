import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SSL_1D(dataset):
    import math

    def toy(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']


        dirichlet_mask = gt['dirichlet_mask']


        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv = jac[:, :, :]

        dvdx = dv[..., 0, 0].squeeze()

        alpha = torch.tensor(1e-7)  # for delta function approx. lim_{\alpha}\goesto zero
        delta = (1 / math.pi) * (alpha / (torch.square(x) + torch.square(alpha)))

        # HJI check
        diff_constraint_hom = dvdx.reshape(-1, 1) - delta
        diff_constraint_hom[:, 0, :] = 0

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[:][dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return toy

def SSL_1D_vh(dataset, alphas):
    import math

    def toy(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        alpha = torch.tensor(alphas)

        dirichlet_mask = gt['dirichlet_mask']


        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv = jac[:, :, :]

        dvdx = dv[..., 0, 0].squeeze()

        # alpha = torch.tensor(1e-6)  # for delta function approx. lim_{\alpha}\goesto zero
        delta = (1 / math.pi) * (alpha / (torch.square(x) + torch.square(alpha)))

        # HJI check
        diff_constraint_hom = dvdx.reshape(-1, 1) - delta
        diff_constraint_hom[:, 0:80, :] = 0

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[:][dirichlet_mask]

        if alphas <= 0.05:
            w = 1e9
        else:
            w = 1

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()/w}

    return toy

def Sup_1D(dataset):
    def toy_s(model_output, gt):
        gt_value = gt['gt_values']
        v_pred = model_output['model_out']
        x = model_output['model_in']
        dvdx, _ = diff_operators.jacobian(v_pred, x)

        return {'values_difference': torch.abs(gt_value - v_pred).sum(),
                'costates_difference': torch.abs(dvdx).sum()}

    return toy_s

def Hybrid_1D(dataset):
    import math
    def toy_h(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']

        gt_values = gt['gt_values']
        sup_index = 2


        dirichlet_mask = gt['dirichlet_mask']


        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv = jac[:, :, :]


        dvdx = dv[..., 0, 0].squeeze()

        alpha = torch.tensor(1e-7)  # for delta function approx. lim_{\alpha}\goesto zero
        delta = (1 / math.pi) * (alpha / (torch.square(x) + torch.square(alpha)))

        # HJI check
        diff_constraint_hom = dvdx.reshape(-1, 1) - delta
        diff_constraint_hom[:, 0, :] = 0


        # boundary condition check
        dirichlet = y[:, :sup_index][dirichlet_mask] - source_boundary_values[:][dirichlet_mask]

        # supervised part
        value_difference = y[:, sup_index:] - gt_values
        costate_difference = dvdx[sup_index:].reshape(-1, 1)


        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'values_difference': torch.abs(value_difference).sum(),
                'costate_difference': torch.abs(costate_difference).sum()}

    return toy_h

