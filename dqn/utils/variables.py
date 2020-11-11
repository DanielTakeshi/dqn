"""
Handle tensors with PyTorch, and creation of other 'data' stuff/items. Note that
this code was originally written using PyTorch 0.3.1, which used the now
deprecated Variable class. See:

    https://pytorch.org/blog/pytorch-0_4_0-migration-guide/

for migration to future versions.

Avoid using `type(x)` on a tensor. Use `x.type()` as that's more specific, e.g.,
`torch.Tensor` vs `torch.FloatTensor`.
"""
import torch
from torch.autograd import Variable


_tensor_convertion = {
    "float": lambda x: x.float(),
    "long": lambda x: x.long(),
    "byte": lambda x: x.byte()
}


_tensor_type_table = {
    "gpu": {
        "float": torch.cuda.FloatTensor,
        "long": torch.cuda.LongTensor,
        "byte": torch.cuda.ByteTensor
    },
    "cpu": {
        "float": torch.FloatTensor,
        "long": torch.LongTensor,
        "byte": torch.ByteTensor
    }
}


_torch_construction_func = {
    "numpy": lambda x: torch.from_numpy(x),
    "torch_cat": lambda x: torch.cat(x),
    "torch_zeros": lambda x: torch.zeros(x),
    "torch_zeros_like": lambda x: torch.zeros_like(x),
    "torch_ones": lambda x: torch.ones(x),
    "torch_ones_minus_eye": lambda x: torch.ones((x, x)) - torch.eye(x)
}


#https://discuss.pytorch.org/t/is-there-any-difference-between-x-to-cuda-vs-x-cuda-which-one-should-i-use/20137
def cuda(x, gpu=True, gpu_id=0, gpu_async=True):
    if gpu:
        with torch.cuda.device(gpu_id):
            if isinstance(x, Variable):
                return x.cuda(async=gpu_async)
            else:
                return x.cuda()
    else:
        return x


def create_tensor(x, gpu=True, gpu_id=0, gpu_async=True, convert_func="numpy"):
    """Creates a tensor.

    Previously, we would use this, then wrap a Variable(...) around the return
    value of this. But it's no longer needed. See above for the various
    construction functions at our disposal.
    """
    x = _torch_construction_func[convert_func](x)
    x = cuda(x, gpu=gpu, gpu_id=gpu_id, gpu_async=gpu_async)
    return x


def create_var(x, gpu=True, gpu_id=0, requires_grad=False,
               gpu_async=True, volatile=False, convert_func="numpy"):
    """Creates a tensor.

    From PyTorch 0.3.1, we would call this from external code, and then this
    wraps around a PyTorch tensor, like this:

        Variable(x, requires_grad=requires_grad, volatile=volatile)

    Now, Variables and Tensors are the same so 'wrapping' is no longer needed.
    Also, volatile is deprecated so we really don't need it. I would say we just
    rely entirely on `requires_grad`?
    """
    with torch.set_grad_enabled(requires_grad):
        x = create_tensor(x=x, gpu=gpu, gpu_id=gpu_id, gpu_async=gpu_async,
                          convert_func=convert_func)
        return x


def create_batch(transitions, gpu=True, gpu_id=0, gpu_async=True, requires_grad=False):
    """Create a batch of data for training, from a `Transition` object.

    From docs: `Previously, any computation that involves a Variable with
    volatile=True wouldn’t be tracked by autograd`. I think that's why Allen put
    volatile=True for the next states, becuase those are tracked by the target
    network, which should not be updated. But, I don't think we use any more ...
    """
    transitions.stack()
    states = create_var(transitions.state, gpu=gpu, gpu_id=gpu_id,
                        requires_grad=requires_grad, gpu_async=gpu_async,
                        convert_func="numpy")
    next_states = create_var(transitions.next_state, gpu=gpu, gpu_id=gpu_id,
                             gpu_async=gpu_async, volatile=True,
                             convert_func="numpy")
    actions = create_var(transitions.action, gpu=gpu, gpu_id=gpu_id,
                         gpu_async=gpu_async, convert_func="numpy")
    rewards = create_var(transitions.reward, gpu=gpu, gpu_id=gpu_id,
                         gpu_async=gpu_async, convert_func="numpy")
    dones = create_var(transitions.done, gpu=gpu, gpu_id=gpu_id,
                       gpu_async=gpu_async, convert_func="numpy")
    return states, next_states, actions, rewards, dones
