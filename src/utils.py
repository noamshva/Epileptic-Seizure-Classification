import math
import os

import numpy as np
import torch

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Connection

# --- tunable classifier hyperparameters (env-overridable for cheap sweeps) ---
# The fully-connected classifier layers were starving the output layer: with
# thresh=10 and init weights ~0.15, the membrane couldn't accumulate to
# threshold before leaking, so output never spiked. Lower threshold and/or
# larger init weight scale let the deep layers fire. Sweep with
# src/diagnose_snn_activity.py (no training needed) before committing to a run.
FC_THRESH = float(os.environ.get("FC_THRESH", "2"))        # classifier LIF threshold
FC_W_SCALE = float(os.environ.get("FC_W_SCALE", "0.3"))    # init weight scale (× rand)


class Resonators2BandsLayer(Network):

    def __init__(self, channel_name, band_name, clk_freq_resonators: torch.Tensor):
        super().__init__(dt=1)
        self.n_resonators = len(clk_freq_resonators)
        self.band_name = band_name
        self.channel_name = channel_name
        # Layers
        input_layer = Input(
            n=self.n_resonators, shape=(self.n_resonators,), traces=True, tc_trace=20.0
        )

        agg_layer = LIFNodes(
            n=1,
            traces=False,
            rest=0.0,
            reset=3 * self.n_resonators,
            thresh=5 * self.n_resonators,
            tc_decay=60 * self.n_resonators,
            refrac=1,
        )
        w = torch.where(clk_freq_resonators == 153600, 1.0, 11.0)
        input_exc_conn = Connection(
            source=input_layer,
            target=agg_layer,
            w=w,
        )
        self.add_layer(input_layer, name=f"{channel_name}-{band_name}-resonators")
        self.add_layer(agg_layer, name=f"{channel_name}-{band_name}-band")

        self.add_connection(
            input_exc_conn,
            source=f"{channel_name}-{band_name}-resonators",
            target=f"{channel_name}-{band_name}-band",
        )


class Bands2TopographyLayer(Network):

    def __init__(
        self, band_name: str, ch_netowrks: dict, shape: tuple, ch_pos: dict, sigma=1
    ):
        super().__init__(dt=1)

        self.ch_netowrks = ch_netowrks
        self.band_name = band_name
        topography_layer = LIFNodes(
            shape=shape,
            n=math.prod(shape),
            traces=True,
            rest=0.0,
            thresh=4,
        )

        # add layers ch_network to topography network!
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            self.add_layer(
                ch_network.layers[f"{ch}-{band_name}-resonators"],
                name=f"{ch}-{band_name}-resonators",
            )
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            self.add_layer(
                ch_network.layers[f"{ch}-{band_name}-band"],
                name=f"{ch}-{band_name}-band",
            )

        self.add_layer(topography_layer, name=f"{band_name}-topography")
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            for (source, target), conn in ch_network.connections.items():
                self.add_connection(conn, source=source, target=target)

            output_ch_layer = ch_network.layers[f"{ch}-{band_name}-band"]
            w = self.channel_weights(shape, center=ch_pos[ch], sigma=sigma)
            conn = Connection(
                source=output_ch_layer,
                target=topography_layer,
                w=w,
            )
            self.add_connection(
                conn, source=f"{ch}-{band_name}-band", target=f"{band_name}-topography"
            )

    def channel_weights(self, kernel_shape, center, sigma=1):
        # Create an empty kernel
        weights = torch.zeros(kernel_shape)

        # Calculate the Gaussian values for the new kernel
        radius = (kernel_shape[0] - 1) / 2
        dr = 1
        xy_center = ((kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2)
        for x in range(kernel_shape[0]):
            for y in range(kernel_shape[1]):
                distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
                weights[x, y] = np.exp(-distance_squared / (2 * sigma**2))

                r = np.sqrt((x - xy_center[0]) ** 2 + (y - xy_center[1]) ** 2)
                if (r - dr / 2) > radius:
                    weights[x, y] = 0

        # Normalize the kernel
        weights /= weights.max()
        # return weights
        return weights.view(1, -1)


class Topographies2SNN(Network):
    """Classifier SNN matching the SA paper (Hadad et al. 2024) layer sizing:

        topography (5 bands × 121 cells)
            ↓  PostPre STDP
        input_layer (121 SCTNs — paper's "input layer", receives 5 × 121 inputs)
            ↓  PostPre STDP
        hidden1_layer (64 SCTNs)
            ↓  PostPre STDP
        hidden2_layer (32 SCTNs)
            ↓  PostPre STDP
        output_layer (2 SCTNs — seizure / non-seizure)
    """

    def __init__(
        self,
        topography_maps: dict[str, Bands2TopographyLayer],
        input_neurons: int = 121,
        hidden1_neurons: int = 64,
        hidden2_neurons: int = 32,
        output_neurons: int = 2,
    ):
        super().__init__()
        self.topography_maps = topography_maps

        input_layer = LIFNodes(
            n=input_neurons,
            traces=True,
            rest=0.0,
            thresh=FC_THRESH,
        )
        self.add_layer(input_layer, name="input_layer")

        hidden1_layer = LIFNodes(
            n=hidden1_neurons,
            traces=True,
            rest=0.0,
            thresh=FC_THRESH,
        )
        self.add_layer(hidden1_layer, name="hidden1_layer")

        hidden2_layer = LIFNodes(
            n=hidden2_neurons,
            traces=True,
            rest=0.0,
            thresh=FC_THRESH,
        )
        self.add_layer(hidden2_layer, name="hidden2_layer")

        output_layer = LIFNodes(
            n=output_neurons,
            traces=True,
            rest=0.0,
            thresh=FC_THRESH,
        )
        self.add_layer(output_layer, name="output_layer")

        # Wire each band's topography into the input_layer.
        for band_name, topography_map in topography_maps.items():
            for layer_name, layer in topography_map.layers.items():
                self.add_layer(layer, name=layer_name)
            for (source, target), conn in topography_map.connections.items():
                self.add_connection(conn, source=source, target=target)

            topography_layer = topography_map.layers[f"{band_name}-topography"]
            w = FC_W_SCALE * torch.rand(topography_layer.n, input_neurons)
            self.add_connection(
                Connection(
                    source=topography_layer,
                    target=input_layer,
                    w=w,
                    update_rule=PostPre,
                ),
                source=f"{band_name}-topography",
                target="input_layer",
            )

        # input_layer → hidden1_layer
        self.add_connection(
            Connection(
                source=input_layer,
                target=hidden1_layer,
                w=FC_W_SCALE * torch.rand(input_neurons, hidden1_neurons),
                update_rule=PostPre,
            ),
            source="input_layer",
            target="hidden1_layer",
        )

        # hidden1_layer → hidden2_layer
        self.add_connection(
            Connection(
                source=hidden1_layer,
                target=hidden2_layer,
                w=FC_W_SCALE * torch.rand(hidden1_neurons, hidden2_neurons),
                update_rule=PostPre,
            ),
            source="hidden1_layer",
            target="hidden2_layer",
        )

        # hidden2_layer → output_layer
        self.add_connection(
            Connection(
                source=hidden2_layer,
                target=output_layer,
                w=FC_W_SCALE * torch.rand(hidden2_neurons, output_neurons),
                update_rule=PostPre,
            ),
            source="hidden2_layer",
            target="output_layer",
        )
