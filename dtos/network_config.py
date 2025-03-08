from typing import NamedTuple

class NetworkConfig(NamedTuple):
    name: str
    channels: list[int]
    kernel_config: str
    resolution: int
    dilation: int
    stride: int