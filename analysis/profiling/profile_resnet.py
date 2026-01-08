# Example of how to use PyTorch profiler to profile a ResNet model.
#
# You can open `trace.json` in Chrome by navigating to
#   chrome://tracing and loading the file for a detailed visualization.

import os

import torch
import torchvision.models as models
from torch.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,  # for TensorBoard support (visualization)
)

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# # Without tensorboard trace handler
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     for i in range(10):
#         model(inputs)
#         prof.step()  # tells profiler when we are doing a new step / iteration

# With tensorboard trace handler
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    on_trace_ready=tensorboard_trace_handler("./log/resnet18"),  # new
) as prof:
    for i in range(10):
        model(inputs)
        prof.step()  # tells profiler when we are doing a new step / iteration


with open("analysis/profiling/profile_resnet.txt", "w") as f:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10), file=f)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30), file=f)
    ## Not needed because done by tensorboard trace handler
    # prof.export_chrome_trace("analysis/profiling/trace.json")
