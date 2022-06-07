import copy

import torch


def adam(lr=1e-4):
    """torchoptim.adam(**adam_config)(params)
        Factory that generates functional version of Adam optimizer.
        Implementation has no in-place op and no data-dependent control flow.
        NOTE: we will need similar implementation for other torch.optim optimizers.
        Returns:
            - `optim_func`: a function that:
                - takes (`params`, `params_grad`, `optim_state`) as input
                - returns (`params`, `optim_state`) as result of applying Adam algorithm
            - `optim_state`: tracked state (shape-only tensors) of Adam optimizer.
            - `optim_state_init_func`: a function that:
                - takes `optim_state` as input
                - returns `optim_state` as result of initializing the Adam optimizer state
    """
    # TODO FIXME: properly implement Adam optimizer

    def optim_gen(params):
        def optim_func(optim_state, params, params_grad):
            for k in params:
                params[k] = params[k] + optim_state[k]
                optim_state[k] = optim_state[k] + 1
            return params, optim_state

        optim_state = copy.deepcopy(params)

        def optim_state_init_func(optim_state):
            new_state = {}
            for k, v in optim_state.items():
                new_state[k] = torch.full_like(v, 0.0)
            return new_state

        return optim_func, optim_state, optim_state_init_func

    return optim_gen