import torch
import tempfile
import uuid
import os
import json
from pathlib import Path
import subprocess

this_file = Path(__file__).resolve()
main_binary = this_file.parent / 'main'

def _full_name(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__

VALID_LOCAL_PARAMS = set(['quat', 'angle'])

class ScriptFunctionCost:
    def __init__(self, fn):
        self.fn = fn
    def forward(*args):
        return self.fn(*args)

class Problem:
    """A nonlinear optimization problem.

    Attributes:
        num_threads: How many threads Ceres will use, both for evaluating the
            Jacobian and solving the linear system. Default is 1.
        max_iterations: Default is 10.
        function_tolerance: Default is float64 epsilon.
        gradient_tolerance: Default is float64 epsilon.
        parameter_tolerance: Default is float64 epsilon.
        linear_solver: One of ('dense_qr', 'sparse_normal_cholesky'). Default is 'dense_qr'.
    """

    def __init__(self):
        self.modules = {}
        self.module_num_residuals = {}
        self.cost_fns = []
        self.params = []
        self.param_map = {}
        self.local_params = {}
        self.num_threads = 1
        self.max_iterations = 10
        self.function_tolerance = torch.finfo(torch.float64).eps
        self.gradient_tolerance = torch.finfo(torch.float64).eps
        self.parameter_tolerance = torch.finfo(torch.float64).eps
        self.linear_solver = 'dense_qr'

    def add_residual(self, cost_fn, *params):
        pidxs = []
        for p in params:
            if p in self.param_map:
                pidx = self.param_map[p]
            else:
                pidx = len(self.params)
                self.params.append(p)
                self.param_map[p] = pidx
            pidxs.append(pidx)

        module_name = _full_name(cost_fn)
        if module_name in self.modules:
            num_residuals = self.module_num_residuals[module_name]
        else:
            self.modules[module_name] = cost_fn
            cost = cost_fn(*params)
            if len(cost.shape) == 0:
                num_residuals = 1
            elif len(cost.shape) == 1:
                num_residuals = cost.shape[0]
            elif len(cost.shape) == 2:
                num_residuals = cost.shape[1]
            else:
                raise ValueError('only 0d or 1d or 2d?? residuals')
            self.module_num_residuals[module_name] = num_residuals

        constant_params = {}
        if hasattr(cost_fn, 'named_parameters'):
            for name, p in cost_fn.named_parameters():
                constant_params[name] = p.tolist()

        self.cost_fns.append({
            'name': type(cost_fn).__name__+'#'+str(uuid.uuid4())[:8],
            'module': module_name,
            'constant_params': constant_params,
            'pidxs': pidxs,
            'num_residuals': num_residuals,
        })

    def _description(self):
        kept_keys = ['name', 'module', 'constant_params', 'pidxs', 'num_residuals']
        cost_fns = [{k: cost_fn[k] for k in kept_keys} for cost_fn in self.cost_fns]
        params = []
        for p in self.params:
            pobj = {}
            if len(p.size()) == 0:
                pobj['values'] = [p.item()]
            elif len(p.size()) == 1:
                pobj['values'] = p.tolist()
            else:
                raise ValueError('only 0d or 1d params')
            params.append(pobj)
        return {
            'cost_fns': cost_fns,
            'params': params,
            'module_names': list(self.modules.keys()),
            'options': {
                'num_threads': self.num_threads,
                'max_iterations': self.max_iterations,
                'function_tolerance': self.function_tolerance,
                'gradient_tolerance': self.gradient_tolerance,
                'parameter_tolerance': self.parameter_tolerance,
                'linear_solver': self.linear_solver,
            },
            'local_params': self.local_params,
        }

    def set_local_parameterization(self, param, name):
        """Sets the local parameterization for a parameter.

        Supported parameterizations:
        
            quat: Quaternion in [w,x,y,z] format

            angle: Angle constrained to [-pi, pi)

        Args:
            name: One of ('quat', 'angle').
        """
        if param not in self.param_map:
            raise RuntimeError('param doesnt exist!')
        pidx = self.param_map[param]
        assert name in VALID_LOCAL_PARAMS
        self.local_params[pidx] = name

    def solve(self, verbose=False, abort=False):
        with tempfile.TemporaryDirectory() as tmp:
            desc = self._description()
            desc['options']['verbose'] = verbose
            with open(os.path.join(tmp, 'description.json'), 'w') as f:
                json.dump(desc, f)
            for module_name, module in self.modules.items():
                fn = os.path.join(tmp, module_name) + '.pt'
                with open(fn, 'wb') as f:
                    torch.jit.save(module, f)

            if abort:
                print('aborting!')
                import shutil
                if os.path.exists('/tmp/torchsolver'):
                    shutil.rmtree('/tmp/torchsolver')
                shutil.copytree(tmp, '/tmp/torchsolver')
                return True

            subprocess.check_call([main_binary, tmp])
            with open(os.path.join(tmp, 'output.json'), 'r') as f:
                output = json.load(f)
        for i, (output_param, param) in enumerate(zip(output, self.params)):
            if len(param.shape) == 0:
                param.view(1)[0] = output_param[0]
            elif len(param.shape) == 1:
                for j in range(param.shape[0]):
                    param[j] = output_param[j]
