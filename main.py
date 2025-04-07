import torch
import argparse
import ast
import os
import operator

from typing import List, Tuple
from random import choice, randint
from torch import nn, fx
from andesPrune.nas.search_space import *


ALL_SEARCH_SPACE = [Conv, Linear, Pooling, BatchNorm, Act, Transpose, Split, Branch]
SEQUENCE_OPS = [Conv, Pooling, BatchNorm, Act, Transpose]
BRANCH_OPS = [Split, Branch]

MAX_WIDTH = 2
MAX_COUNT = 50
COUNT = 0


def grow_down(
    curr_input_nodes: List[Tuple[fx.Node, int]],
    curr_output_nodes: List[Tuple[fx.Node, int]],
    model: nn.Module,
    graph: fx.Graph,
):
    global COUNT
    CURR_SEARCH_SPACE = []
    node, output_idx = curr_output_nodes.pop(0)
    compatible_nodes = []
    for OP in ALL_SEARCH_SPACE:
        if OP == Add:
            for node2, output_idx2 in curr_output_nodes:
                if Add.check(
                    input_tensors=[
                        model.get_submodule(node.target).output_tensors[output_idx],
                        model.get_submodule(node2.target).output_tensors[output_idx2],
                    ],
                    grow_up=False,
                ):
                    CURR_SEARCH_SPACE.append(OP)
                    compatible_nodes.append((node2, output_idx2))

        elif OP == Branch or OP == Split:
            if len(curr_output_nodes) < MAX_WIDTH and OP.check(
                input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False
            ):
                CURR_SEARCH_SPACE.append(OP)

        else:
            if OP.check(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False):
                CURR_SEARCH_SPACE.append(OP)

    OP = choice(CURR_SEARCH_SPACE)
    if OP == Conv:
        op = Conv(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Linear:
        op = Linear(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Pooling:
        op = Pooling(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == BatchNorm:
        op = BatchNorm(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Act:
        op = Act(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Transpose:
        op = Transpose(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Add:
        comp_node, comp_idx = compatible_nodes.pop(randint(0, len(compatible_nodes) - 1))
        op = Add(
            input_tensors=[
                model.get_submodule(node.target).output_tensors[output_idx],
                model.get_submodule(comp_node.target).output_tensors[comp_idx],
            ],
            grow_up=False,
        )
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node, comp_node))
        curr_output_nodes.append((insert_node, 0))

    elif OP == Split or OP == Branch:
        op = OP(input_tensors=[model.get_submodule(node.target).output_tensors[output_idx]], grow_up=False)
        setattr(model, f"{op}_{COUNT}", op)
        insert_node = graph.call_module(f"{op}_{COUNT}", args=(node,))
        node.append(insert_node)
        cur_node = insert_node
        for i in range(op.num):
            COUNT += 1
            op = Getitem(input_tensors=model.get_submodule(insert_node.target).output_tensors, grow_up=False, idx=i)
            setattr(model, f"{op}_{COUNT}", op)
            getitem_node = graph.call_module(f"{op}_{COUNT}", args=(insert_node,))
            cur_node.append(getitem_node)
            cur_node = getitem_node
            curr_output_nodes.append((getitem_node, 0))

    else:
        raise RuntimeError("")


def grow_up(
    curr_input_nodes: List[Tuple[fx.Node, int]],
    curr_output_nodes: List[Tuple[fx.Node, int]],
    model: nn.Module,
    graph: fx.Graph,
    COUNT: int,
):
    pass


def build_graph(model: nn.Module, graph: fx.Graph, to_onnx: bool = False, onnx_path: str = "gen.onnx"):
    graph.lint()
    gm = fx.GraphModule(model, graph)

    input = []
    for name, m in gm.named_modules():
        if "input" in name:
            input += m.input_tensors
    gm(*input)
    print(gm)
    if to_onnx:
        import netron
        import onnx
        import onnxsim

        torch.onnx.export(gm, tuple(input), onnx_path)
        onnx_model = onnx.load(onnx_path)
        model_simp, check = onnxsim.simplify(onnx_model)
        onnx.save(model_simp, onnx_path)
        netron.start(onnx_path)
        breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AndesTech network compression")
    parser.add_argument("--input_shapes", type=str, nargs="+", default=["(1, 3, 32, 32)"], help="shape of input tensors")
    parser.add_argument(
        "--input_layouts",
        type=str,
        nargs="+",
        default=["nchw"],
        choices=["nc", "ncl", "nchw", "ncdhw"],
        help="layout of input tensors",
    )
    args = parser.parse_args()

    # Parse the string representation of the list into a Python list
    input_tensors = []
    for shape, layout in zip(args.input_shapes, args.input_layouts):
        tensor = torch.rand(ast.literal_eval(shape))
        tensor.LAYOUT = layout
        input_tensors.append(tensor)

    for _ in range(100):
        model = Root()
        graph = fx.Graph()
        COUNT = 0

        curr_input_nodes = []
        curr_output_nodes = []
        for i, input_tensor in enumerate(input_tensors):
            input_node = graph.placeholder(f"x_{i}")
            module = Placeholder([input_tensor])
            setattr(model, f"input_module_{i}", module)
            with graph.inserting_after(input_node):
                input_module_node = graph.call_module(f"input_module_{i}", args=(input_node,))
            curr_output_nodes.append((input_module_node, 0))

        while COUNT < MAX_COUNT and (curr_input_nodes or curr_output_nodes):
            grow_down(curr_input_nodes, curr_output_nodes, model, graph)
            COUNT += 1

        output_nodes = []
        for node, idx in curr_output_nodes:
            op = Output()
            setattr(model, f"{op}_{COUNT}", op)
            output_nodes.append(graph.call_module(f"{op}_{COUNT}", args=(node,)))
        output_node = graph.output(tuple(output_nodes))
        build_graph(model, graph, True)
