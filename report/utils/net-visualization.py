import sys
from PlotNeuralNet.PyCore.TikzGen import (
    ToBegin,
    ToConnection,
    ToCor,
    ToEnd,
    ToGenerate,
    ToHead,
    ToInput,
    ToConv,
    ToSum,
    ToSoftMax,
    ToFullyConnected
)

arch = \
[
    ToHead(".."),
    ToCor(),
    ToBegin(),
    ToInput("ax.png", to="(0, 4, 0)", width=3, height=3, name="in_ax"),
    ToInput("front.png", to="(0, 0, 0)", width=3, height=3, name="in_front"),
    ToInput("sag.png", to="(0, -4, 0)", width=3, height=3, name="in_sag"),
    
    ToConv("conv_ax", 1024, 1, offset="(2, 0, 0)", to="(in_ax)", width=8, height=20, depth=20, caption="ConvNeXt Ax"),
    ToConv("conv_front", 1024, 1, offset="(2, 0, 0)", to="(in_front)", width=8, height=20, depth=20, caption="ConvNeXt Front"),
    ToConv("conv_sag", 1024, 1, offset="(2, 0, 0)", to="(in_sag)", width=8, height=20, depth=20, caption="ConvNeXt Sag"),
    
    ToConnection("in_ax", "conv_ax"),
    ToConnection("in_front", "conv_front"),
    ToConnection("in_sag", "conv_sag"),
    
    ToConv("stack_kv", 1024, 3, offset="(2, 0, 0)", to="(conv_front-east)", width=2, height=20, depth=20, caption="Stack (KV)"),
    ToConnection("conv_ax", "stack_kv"),
    ToConnection("conv_front", "stack_kv"),
    ToConnection("conv_sag", "stack_kv"),
    
    ToFullyConnected("mha_ax", 1024, offset="(3, 0, 0)", to="(conv_ax-east)", width=4, height=20, depth=5, caption="MHA Ax"),
    ToFullyConnected("mha_front", 1024, offset="(3, 0, 0)", to="(conv_front-east)", width=4, height=20, depth=5, caption="MHA Front"),
    ToFullyConnected("mha_sag", 1024, offset="(3, 0, 0)", to="(conv_sag-east)", width=4, height=20, depth=5, caption="MHA Sag"),
    
    ToConnection("conv_ax", "mha_ax"),
    ToConnection("conv_front", "mha_front"),
    ToConnection("conv_sag", "mha_sag"),
    
    ToConnection("stack_kv", "mha_ax"),
    ToConnection("stack_kv", "mha_front"),
    ToConnection("stack_kv", "mha_sag"),
    
    ToConv("stack_attn", 1024, 3, offset="(2, 0, 0)", to="(mha_front-east)", width=2, height=20, depth=20, caption="Stack (Attn)"),
    ToConnection("mha_ax", "stack_attn"),
    ToConnection("mha_front", "stack_attn"),
    ToConnection("mha_sag", "stack_attn"),
    
    ToSum("fusion", offset="(2, 0, 0)", to="(stack_attn-east)", radius=3, opacity=0.6),
    ToConnection("stack_attn", "fusion"),
    
    ToFullyConnected("fc1", 1024, offset="(3, 0, 0)", to="(fusion-east)", width=4, height=15, depth=2, caption="Linear + ReLU"),
    ToConnection("fusion", "fc1"),
    
    ToFullyConnected("fc2", 64, offset="(3, 0, 0)", to="(fc1-east)", width=4, height=15, depth=2, caption="Linear"),
    ToConnection("fc1", "fc2"),
    
    ToSoftMax("soft1", 6, "(3, 0, 0)", to="(fc2-east)", width=1, height=6, depth=8, caption="Softmax"),
    ToConnection("fc2", "soft1"),
    
    ToEnd(),
]


if __name__ == "__main__":
    nameFile = str(sys.argv[0]).split(".")[0]
    ToGenerate(arch, nameFile + ".tex")
