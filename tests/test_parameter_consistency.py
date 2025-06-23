#!/usr/bin/env python3
"""
Demo script showing that activation and transform parameters now have identical capabilities.
"""

import torch
from src.skagent.ann import FlexibleNet


def custom_activation(x):
    """Custom activation function: leaky ReLU with slope 0.1"""
    return torch.where(x > 0, x, 0.1 * x)


def custom_transform(x):
    """Custom transform function: scale and shift"""
    return 2 * x + 1


print("=== FlexibleNet Parameter Consistency Demo ===\n")

# Both parameters now support the same capabilities:
examples = [
    ("String", "relu", "sigmoid"),
    ("List", ["relu", "tanh", "silu"], ["sigmoid", "exp", "tanh"]),
    ("Callable", custom_activation, custom_transform),
    ("None/Identity", None, None),
    (
        "Mixed List",
        ["relu", custom_activation, None],
        ["sigmoid", custom_transform, None],
    ),
]

for name, activation, transform in examples:
    print(f"--- {name} Example ---")

    try:
        net = FlexibleNet(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation=activation,
            transform=transform,
        )

        # Test with dummy input on the same device as network
        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        print(f"✅ Activation: {activation}")
        print(f"✅ Transform: {transform}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    except Exception as e:
        print(f"❌ Error: {e}")

    print()

print("🎉 Both activation and transform parameters now support:")
print("   • str: Built-in function names")
print("   • list: Different functions per layer/output")
print("   • callable: Custom functions")
print("   • None: Identity function")
print("   • Full parameter consistency achieved!")
