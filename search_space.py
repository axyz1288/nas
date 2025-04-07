import torch
from typing import List, Tuple
from torch import nn, Tensor
from random import randint, choice, sample
from math import ceil


__all__ = [
    "Root",
    "Placeholder",
    "Output",
    "Conv",
    "Linear",
    "Pooling",
    "BatchNorm",
    "Act",
    "Transpose",
    "Add",
    "Split",
    "Branch",
    "Getitem",
]


MAX_VALUE = 1024
MAX_BRANCH = 6


def common_factors(a, b):
    max_val = min(a, b)  # 兩數的最大可能公因數
    candidates = torch.arange(1, max_val + 1)  # 產生 1 到 max_val 的數列
    mask = (torch.remainder(a, candidates) == 0) & (torch.remainder(b, candidates) == 0)  # 篩選公因數
    return candidates[mask].tolist()  # 只保留公因數


def clone(tensor: Tensor, LAYOUT: str) -> Tensor:
    out = tensor.clone()
    out.LAYOUT = LAYOUT
    return out


class Root(nn.Module):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Root"

    def forward(self, x: Tensor):
        return x

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        return True


class Node(nn.Module):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if grow_up == False:
            assert isinstance(input_tensors, list), f"input_tensors should be given when trace_down, got {input_tensors}"
            for tensor in input_tensors:
                assert isinstance(tensor, Tensor), f"{self.__class__} input_tensor must be list of tensors, got {input_tensors}"
        elif grow_up == True:
            assert isinstance(input_tensors, list), f"output_tensors should be given when grow_up, got {output_tensors}"
            for tensor in input_tensors:
                assert isinstance(tensor, Tensor), f"{self.__class__} input_tensor must be list of tensors, got {output_tensors}"

        super().__init__()
        self.input_tensors: List[Tensor] = input_tensors
        self.output_tensors: List[Tensor] = output_tensors

    def __str__(self):
        raise NotImplementedError

    def forward(self, x: Tensor):
        raise NotImplementedError

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        raise NotImplementedError


class Placeholder(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        super().__init__(input_tensors, output_tensors, grow_up)
        self.output_tensors = [clone(ten, ten.LAYOUT) for ten in self.input_tensors]

    def __str__(self):
        return "Placeholder"

    def forward(self, x: Tensor):
        return x

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        return True


class Output(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        super().__init__(input_tensors, output_tensors, grow_up)
        self.output_tensors = [clone(ten, ten.LAYOUT) for ten in self.input_tensors]

    def __str__(self):
        return "Output"

    def forward(self, x: Tensor):
        return x

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        return True


class Conv(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Convolution only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Convolution only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        M = choice(self.SPACE)
        self.in_channels = x.shape[1]
        self.out_channels = randint(1, MAX_VALUE)
        if x.LAYOUT == "ncl":
            self.kernel_size = randint(1, ceil(x.shape[2] / 2))
            self.stride = randint(1, ceil(self.kernel_size / 2))
            self.padding = randint(0, x.shape[2] - 1)
            self.dilation = randint(1, min(x.shape[2] // self.kernel_size, 1))
        elif x.LAYOUT == "nchw":
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
            )
            self.padding = (
                randint(0, x.shape[2] - 1),
                randint(0, x.shape[3] - 1),
            )
            self.dilation = (
                randint(1, min(x.shape[2] // self.kernel_size[0], 1)),
                randint(1, min(x.shape[3] // self.kernel_size[1], 1)),
            )
        elif x.LAYOUT == "ncdhw":
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
                randint(1, ceil(x.shape[4] / 2)),
            )
            self.stride = self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
                randint(1, ceil(self.kernel_size[2] / 2)),
            )
            self.padding = (
                randint(0, x.shape[2] - 1),
                randint(0, x.shape[3] - 1),
                randint(0, x.shape[4] - 1),
            )
            self.dilation = (
                randint(1, min(x.shape[2] // self.kernel_size[0], 1)),
                randint(1, min(x.shape[3] // self.kernel_size[1], 1)),
                randint(1, min(x.shape[4] // self.kernel_size[2], 1)),
            )

        self.groups = choice(common_factors(self.in_channels, self.out_channels))
        self.bias = choice([True, False])
        self.padding_mode = choice(["zeros"])
        self.module = M(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
        )
        self.module = self.module.eval().to(x.device)
        self.output_tensors = [clone(self.module(x), x.LAYOUT)]

    def __str__(self):
        return "Conv"

    def forward(self, x: Tensor):
        y = self.module(x)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x = input_tensors[0]
            dim = x.dim()

            if dim == 3 and x.LAYOUT == "ncl" and x.shape[2] > 1:
                cls.SPACE = [nn.Conv1d]
                return True
            elif dim == 4 and x.LAYOUT == "nchw" and min(x.shape[2], x.shape[3]) > 1:
                cls.SPACE = [nn.Conv2d]
                return True
            elif dim == 5 and x.LAYOUT == "ncdhw" and min(x.shape[2], x.shape[3], x.shape[4]) > 1:
                cls.SPACE = [nn.Conv3d]
                return True
            else:
                return False
        else:
            raise NotImplementedError


class Linear(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Linear only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Linear only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        self.in_features = x.shape[-1]
        self.out_features = randint(1, MAX_VALUE)
        self.bias = choice([True, False])
        self.module = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias)
        self.module = self.module.eval().to(x.device)
        self.output_tensors = [clone(self.module(x), x.LAYOUT)]

    def __str__(self):
        return "Linear"

    def forward(self, x: Tensor):
        y = self.module(x)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x = input_tensors[0]
            dim = x.dim()
            if dim == 1:
                return False
            else:
                return True
        else:
            raise NotImplementedError


class Pooling(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Pooling only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Pooling only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        M = choice(self.SPACE)
        if M == nn.MaxPool1d:
            self.kernel_size = randint(1, ceil(x.shape[2] / 2))
            self.stride = randint(1, ceil(self.kernel_size / 2))
            self.padding = randint(0, int(self.kernel_size / 2))
            self.dilation = randint(1, min(x.shape[2] // self.kernel_size, 1))
            self.module = nn.MaxPool1d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif M == nn.MaxPool2d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
            )
            self.dilation = (
                randint(1, min(x.shape[2] // self.kernel_size[0], 1)),
                randint(1, min(x.shape[3] // self.kernel_size[1], 1)),
            )
            self.module = nn.MaxPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif M == nn.MaxPool3d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
                randint(1, ceil(x.shape[4] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
                randint(1, ceil(self.kernel_size[2] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
                randint(0, int(self.kernel_size[2] / 2)),
            )
            self.dilation = (
                randint(1, min(x.shape[2] // self.kernel_size[0], 1)),
                randint(1, min(x.shape[3] // self.kernel_size[1], 1)),
                randint(1, min(x.shape[4] // self.kernel_size[2], 1)),
            )
            self.module = nn.MaxPool3d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        elif M == nn.MaxUnpool1d:
            self.kernel_size = randint(1, ceil(x.shape[2] / 2))
            self.stride = randint(1, ceil(self.kernel_size / 2))
            self.padding = randint(0, ceil(self.kernel_size / 2))
            self.module = nn.MaxUnpool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif M == nn.MaxUnpool2d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
            )
            self.module = nn.MaxUnpool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif M == nn.MaxUnpool3d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
                randint(1, ceil(x.shape[4] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
                randint(1, ceil(self.kernel_size[2] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
                randint(0, int(self.kernel_size[2] / 2)),
            )
            self.module = nn.MaxUnpool3d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        elif M == nn.AvgPool1d:
            self.kernel_size = randint(1, ceil(x.shape[2] / 2))
            self.stride = randint(1, ceil(self.kernel_size / 2))
            self.padding = randint(0, ceil(self.kernel_size / 2))
            self.module = nn.LPPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif M == nn.AvgPool2d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
            )
            self.module = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif M == nn.AvgPool3d:
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
                randint(1, ceil(x.shape[4] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
                randint(1, ceil(self.kernel_size[2] / 2)),
            )
            self.padding = (
                randint(0, int(self.kernel_size[0] / 2)),
                randint(0, int(self.kernel_size[1] / 2)),
                randint(0, int(self.kernel_size[2] / 2)),
            )
            self.module = nn.AvgPool3d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        elif M == nn.LPPool1d:
            self.norm_type = choice([1, 2])
            self.kernel_size = randint(1, ceil(x.shape[2] / 2))
            self.stride = randint(1, ceil(self.kernel_size / 2))
            self.module = nn.LPPool1d(norm_type=self.norm_type, kernel_size=self.kernel_size, stride=self.stride)
        elif M == nn.LPPool2d:
            self.norm_type = choice([1, 2])
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
            )
            self.module = nn.LPPool2d(norm_type=self.norm_type, kernel_size=self.kernel_size, stride=self.stride)
        elif M == nn.LPPool3d:
            self.norm_type = choice([1, 2])
            self.kernel_size = (
                randint(1, ceil(x.shape[2] / 2)),
                randint(1, ceil(x.shape[3] / 2)),
                randint(1, ceil(x.shape[4] / 2)),
            )
            self.stride = (
                randint(1, ceil(self.kernel_size[0] / 2)),
                randint(1, ceil(self.kernel_size[1] / 2)),
                randint(1, ceil(self.kernel_size[2] / 2)),
            )
            self.module = nn.LPPool3d(norm_type=self.norm_type, kernel_size=self.kernel_size, stride=self.stride)

        elif M == nn.AdaptiveMaxPool1d:
            self.output_size = randint(1, x.shape[2])
            self.module = nn.AdaptiveMaxPool1d(output_size=self.output_size)
        elif M == nn.AdaptiveMaxPool2d:
            self.output_size = (randint(1, x.shape[2]), randint(1, x.shape[3]))
            self.module = nn.AdaptiveMaxPool2d(output_size=self.output_size)
        elif M == nn.AdaptiveMaxPool3d:
            self.output_size = (randint(1, x.shape[2]), randint(1, x.shape[3]), randint(1, x.shape[4]))
            self.module = nn.AdaptiveMaxPool3d(output_size=self.output_size)

        elif M == nn.AdaptiveAvgPool1d:
            self.output_size = randint(1, x.shape[2])
            self.module = nn.AdaptiveAvgPool1d(output_size=self.output_size)
        elif M == nn.AdaptiveAvgPool2d:
            self.output_size = (randint(1, x.shape[2]), randint(1, x.shape[3]))
            self.module = nn.AdaptiveAvgPool2d(output_size=self.output_size)
        elif M == nn.AdaptiveAvgPool3d:
            self.output_size = (randint(1, x.shape[2]), randint(1, x.shape[3]), randint(1, x.shape[4]))
            self.module = nn.AdaptiveAvgPool3d(output_size=self.output_size)

        self.module = self.module.eval().to(x.device)
        self.output_tensors = [clone(self.module(x), x.LAYOUT)]

    def __str__(self):
        return "Pooling"

    def forward(self, x: Tensor):
        y = self.module(x)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x = input_tensors[0]
            dim = x.dim()

            if dim == 3 and x.LAYOUT == "ncl" and x.shape[2] > 1:
                cls.SPACE = [
                    nn.MaxPool1d,
                    # nn.MaxUnpool1d,
                    nn.AvgPool1d,
                    # nn.LPPool1d,
                    # nn.AdaptiveMaxPool1d,
                    # nn.AdaptiveAvgPool1d,
                ]
                return True
            elif dim == 4 and x.LAYOUT == "nchw" and max(x.shape[2], x.shape[3]) > 1:
                cls.SPACE = [
                    nn.MaxPool2d,
                    # nn.MaxUnpool2d,
                    nn.AvgPool2d,
                    # nn.LPPool2d,
                    # nn.AdaptiveMaxPool2d,
                    # nn.AdaptiveAvgPool2d,
                ]
                return True
            elif dim == 5 and x.LAYOUT == "ncdhw" and max(x.shape[2], x.shape[3], x.shape[4]) > 1:
                cls.SPACE = [
                    nn.MaxPool3d,
                    # nn.MaxUnpool3d,
                    nn.AvgPool3d,
                    nn.LPPool3d,
                    # nn.AdaptiveMaxPool3d,
                    # nn.AdaptiveAvgPool3d,
                ]
                return True
            else:
                return False
        else:
            raise NotImplementedError


class BatchNorm(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"BatchNorm only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"BatchNorm only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        M = choice(self.SPACE)
        x = self.input_tensors[0]
        self.num_features = x.shape[1]
        self.module = M(num_features=self.num_features).eval().to(x.device)
        self.output_tensors = [clone(self.module(x), x.LAYOUT)]

    def __str__(self):
        return "BatchNorm"

    def forward(self, x: Tensor):
        y = self.module(x)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x = input_tensors[0]
            dim = x.dim()

            if dim == 3 and x.LAYOUT == "ncl":
                cls.SPACE = [nn.BatchNorm1d]
                return True
            elif dim == 4 and x.LAYOUT == "nchw":
                cls.SPACE = [nn.BatchNorm2d]
                return True
            elif dim == 5 and x.LAYOUT == "ncdhw":
                cls.SPACE = [nn.BatchNorm3d]
                return True
            else:
                return False
        else:
            raise NotImplementedError


class Act(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Act only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Act only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        M = choice(self.SPACE)
        all = ((torch.rand(1) * 2 - 1) * MAX_VALUE).item()
        left = ((torch.rand(1) - 1) * MAX_VALUE).item()
        right = (torch.rand(1) * MAX_VALUE).item()
        if M == nn.ELU:
            self.module = nn.ELU(alpha=all)
        elif M == nn.Hardshrink:
            self.module = nn.Hardshrink(lambd=all)
        elif M == nn.Hardsigmoid:
            self.module = nn.Hardsigmoid()
        elif M == nn.Hardtanh:
            self.module = nn.Hardtanh(min_value=left, max_value=right)
        elif M == nn.Hardswish:
            self.module = nn.Hardswish()
        elif M == nn.LeakyReLU:
            self.module = nn.LeakyReLU(negative_slope=all)
        elif M == nn.LogSigmoid:
            self.module = nn.LogSigmoid()
        elif M == nn.PReLU:
            self.module = nn.PReLU(num_parameters=x.shape[1], init=all)
        elif M == nn.ReLU:
            self.module = nn.ReLU()
        elif M == nn.ReLU6:
            self.module = nn.ReLU6()
        elif M == nn.RReLU:
            self.module = nn.RReLU(lower=left, upper=right)
        elif M == nn.SELU:
            self.module = nn.SELU()
        elif M == nn.CELU:
            self.module = nn.CELU(alpha=all)
        elif M == nn.GELU:
            self.module = nn.GELU()
        elif M == nn.Sigmoid:
            self.module = nn.Sigmoid()
        elif M == nn.SiLU:
            self.module = nn.SiLU()
        elif M == nn.Mish:
            self.module = nn.Mish()
        elif M == nn.Softplus:
            self.module = nn.Softplus(beta=all, threshold=all)
        elif M == nn.Softshrink:
            self.module = nn.Softshrink(lambd=right)
        elif M == nn.Softsign:
            self.module = nn.Softsign()
        elif M == nn.Tanh:
            self.module = nn.Tanh()
        elif M == nn.Tanhshrink:
            self.module = nn.Tanhshrink()
        elif M == nn.Threshold:
            self.module = nn.Threshold(threshold=all, value=all)
        elif M == nn.GLU:
            dims = []
            for dim in x.shape:
                if dim % 2 == 0:
                    dims.append(dim)
            dim = choice(dims)
            dim = list(x.shape).index(dim)
            self.module = nn.GLU(dim=dim)
        elif M == nn.Softmin:
            self.module = nn.Softmin(dim=choice(list(range(x.dim()))))
        elif M == nn.Softmax:
            self.module = nn.Softmax(dim=choice(list(range(x.dim()))))
        elif M == nn.Softmax2d:
            self.module = nn.Softmax2d()
        elif M == nn.LogSoftmax:
            self.module = nn.LogSoftmax(dim=choice(list(range(x.dim()))))
        else:
            raise RuntimeError("Cannot randomly generate an activation module")

        self.module = self.module.eval().to(x.device)
        self.output_tensors = [clone(self.module(x), x.LAYOUT)]

    def __str__(self):
        return "Act"

    def forward(self, x: Tensor):
        y = self.module(x)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        cls.SPACE = [
            # nn.ELU,
            # nn.Hardshrink,
            # nn.Hardsigmoid,
            # nn.Hardtanh,
            # nn.Hardswish,
            nn.LeakyReLU,
            # nn.LogSigmoid,
            nn.PReLU,
            nn.ReLU,
            nn.ReLU6,
            # nn.RReLU,
            # nn.SELU,
            # nn.CELU,
            # nn.GELU,
            nn.Sigmoid,
            nn.SiLU,
            # nn.Mish,
            # nn.Softplus,
            # nn.Softshrink,
            # nn.Softsign,
            nn.Tanh,
            # nn.Tanhshrink,
            # nn.Threshold,
            # nn.Softmin,
            nn.Softmax,
            nn.LogSoftmax,
        ]
        if not grow_up:
            x = input_tensors[0]
            if any(dim % 2 == 0 for dim in x.shape):
                cls.SPACE.append(nn.GLU)
            if x.LAYOUT == "nchw":
                cls.SPACE.append(nn.Softmax2d)
        else:
            x = output_tensors[0]
            cls.SPACE.append(nn.GLU)
            if x.LAYOUT == "nchw":
                cls.SPACE.append(nn.Softmax2d)
        return True


class Transpose(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Transpose only has one input, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Transpose only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        dims = list(range(x.dim()))
        self.dim0, self.dim1 = sample(dims, k=2)
        self.output_tensors = [torch.transpose(x, self.dim0, self.dim1)]
        self.output_tensors[0].LAYOUT = self._swap_chars(x.LAYOUT, self.dim0, self.dim1)

    def _swap_chars(self, word: str, i: int, j: int):
        if i == j:
            return word  # No need to swap if indices are the same

        word_list = list(word)  # Convert string to a list for mutability
        word_list[i], word_list[j] = word_list[j], word_list[i]  # Swap the characters

        return "".join(word_list)

    def __str__(self):
        return "Transpose"

    def forward(self, x: Tensor):
        y = torch.transpose(x, self.dim0, self.dim1)
        y.LAYOUT = self._swap_chars(x.LAYOUT, self.dim0, self.dim1)
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        x = input_tensors[0] if not grow_up else output_tensors[0]
        if x.dim() == 1:
            return False
        return True


class Add(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Add only has two inputs, got {len(input_tensors)} inputs."
        else:
            assert len(output_tensors) == 1, f"Add only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        x, x_1 = self.input_tensors[0], self.input_tensors[1]
        self.output_tensors = [clone(torch.add(x, x_1), x.LAYOUT)]

    def __str__(self):
        return "Add"

    def forward(self, x: Tensor, x_1: Tensor):
        y = torch.add(x, x_1)
        y.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x, x_1 = input_tensors[0], input_tensors[1]
            if x.LAYOUT == x_1.LAYOUT and x.shape == x_1.shape:
                return True
            else:
                return False
        else:
            raise NotImplementedError


class Split(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Split only has one input, got {len(input_tensors)} inputs."
        else:
            pass
        super().__init__(input_tensors, output_tensors, grow_up)
        x = self.input_tensors[0]
        dims = []
        for dim in range(1, x.dim()):
            if x.size(dim) != 1:
                dims.append(dim)
        self.dim = sample(dims, k=1)[0]
        self.sections = randint(1, min(MAX_BRANCH, x.size(self.dim)))
        for y in torch.split(x, split_size_or_sections=self.sections, dim=self.dim):
            y.LAYOUT = x.LAYOUT
            self.output_tensors.append(y)
        self.num = len(self.output_tensors)

    def __str__(self):
        return "Split"

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = torch.split(x, split_size_or_sections=self.sections, dim=self.dim)
        for tensor in y:
            tensor.LAYOUT = x.LAYOUT
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            x = input_tensors[0]
            if all(dim == 1 for dim in x.shape):
                return False
            return True
        else:
            raise NotImplementedError


class Branch(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            assert len(input_tensors) == 1, f"Convolution only has one input, got {len(input_tensors)} inputs."
        else:
            pass
        super().__init__(input_tensors, output_tensors, grow_up)
        self.num = randint(2, MAX_BRANCH)
        self.output_tensors = [clone(self.input_tensors[0], self.input_tensors[0].LAYOUT) for _ in range(self.num)]

    def __str__(self):
        return "Branch"

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = []
        for _ in range(self.num):
            y.append(clone(x, x.LAYOUT))
        return y

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        if not grow_up:
            return True
        else:
            return False


class Getitem(Node):
    def __init__(
        self, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False, *args, **kwargs
    ):
        if not grow_up:
            pass
        else:
            assert len(output_tensors) == 1, f"Convolution only has one output, got {len(output_tensors)} outputs."
        super().__init__(input_tensors, output_tensors, grow_up)
        self.idx = kwargs["idx"]
        self.output_tensors = [input_tensors[self.idx]]

    def __str__(self):
        return "Getitem"

    def forward(self, x: Tuple[Tensor]) -> Tuple[Tensor, ...]:
        return x[self.idx]

    @classmethod
    def check(cls, input_tensors: List[Tensor] = [], output_tensors: List[Tensor] = [], grow_up: bool = False):
        return True
