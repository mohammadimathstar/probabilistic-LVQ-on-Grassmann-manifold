import argparse
from xml.parsers.expat import model
import numpy as np
import os
import torch
import torch.nn as nn
from util.grassmann import grassmann_repr, grassmann_repr_full
from lvq.prototypes import PrototypeLayer
from lvq.measures.base_measure import GrassmannMeasureBase


LOW_BOUND_LAMBDA = 0.001


class GrassmannLVQModel(nn.Module):
    """
    A Grassmann-based LVQ model with a feature extractor, optional add-on layers,
    and a prototype layer that operates on subspace representations.
    """
    def __init__(
        self,
        num_classes: int,
        feature_extractor: torch.nn.Module,
        score_fn: GrassmannMeasureBase,
        args: argparse.Namespace,
        add_on_layers: nn.Module = nn.Identity(),
        device: str = 'cpu',
        init_from_data: bool = False,
        dataloader=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.add_on_layers = add_on_layers

        # Create the prototype layer
        self.prototype_layer = PrototypeLayer(
            num_classes=self.num_classes,
            score_fn=score_fn,
            args=args,
            device=device,
        )

        self.to(device)

        # Optionally initialize prototypes from real data
        if init_from_data and dataloader is not None:
            self.initialize_prototypes_from_data(dataloader, device)

        # Check if prototypes require gradients
        print(self.prototype_layer.xprotos.requires_grad)

        



    # -------------------------------------------------------------------------
    # Require grad toggles
    # -------------------------------------------------------------------------
    def set_requires_grad(self, module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad = requires_grad

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.xprotos.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.xprotos.requires_grad = val

    # -------------------------------------------------------------------------
    # Forward functions
    # -------------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, **kwargs):
        """
        Forward pass of the model.
        Args:
            inputs (torch.Tensor): A batch of input data.
        Returns:
            torch.Tensor: Distance between input subspaces and prototypes.
        """
        features = self.feature_extractor(inputs)
        features = self.add_on_layers(features)

        subspaces = grassmann_repr(features, self.prototype_layer.subspace_dim)
        scores = self.prototype_layer(subspaces)
        return scores

    def forward_partial(self, inputs: torch.Tensor):
        """
        Forward pass returning intermediate representations.
        """
        features = self.feature_extractor(inputs)
        features = self.add_on_layers(features)

        subspaces, Vh, S = grassmann_repr_full(features, self.prototype_layer.subspace_dim)

        output = self.prototype_layer.score_fn(
            subspaces,
            self.prototype_layer.xprotos,
            relevances=self.prototype_layer.relevances,
        )

        return features, subspaces, Vh, S, output

    # -------------------------------------------------------------------------
    # Prototype initialization from real data
    # -------------------------------------------------------------------------
    def initialize_prototypes_from_data(self, dataloader, device):
        """
        Initializes the prototypes using one random sample per class from the dataloader.
        """
        print("Initializing prototypes from real data...")
        self.eval()
        selected = {}
        labels = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            for x, y in zip(inputs, targets):
                y_int = int(y.item())
                if y_int not in selected:
                    selected[y_int] = x.unsqueeze(0)
                    labels.append(y_int)
                if len(selected) == self.num_classes:
                    break
            if len(selected) == self.num_classes:
                break

        if len(selected) < self.num_classes:
            raise ValueError(
                f"Not enough distinct classes found: {len(selected)} / {self.num_classes}"
            )

        imgs = torch.cat(list(selected.values()), dim=0).to(device)
        with torch.no_grad():
            features = self.add_on_layers(self.feature_extractor(imgs))
            subspaces = grassmann_repr(features, self.prototype_layer.subspace_dim)

        self.prototype_layer.xprotos.data = subspaces.clone()
        self.prototype_layer.yprotos = torch.tensor(labels, dtype=torch.int32, device=device)
        print("✅ Prototypes initialized from real samples.")

    # -------------------------------------------------------------------------
    # Save / load methods
    # -------------------------------------------------------------------------
    def save_state(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        torch.save(self.state_dict(), f"{directory_path}/model_state.pth")

    @staticmethod
    def load_state(directory_path: str, model_class, *args, **kwargs):
        model = model_class(*args, **kwargs)
        model.load_state_dict(
            torch.load(f"{directory_path}/model_state.pth", map_location="cpu")
        )
        return model


def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train acc: {f['accuracy_of_train_set'][-1]}, "
              f"val acc: {f['accuracy_of_validation_set'][-1]} "
              f"(max={np.max(f['accuracy_of_validation_set'])})")
    return xprotos, yprotos, lamda






# import argparse
# import numpy as np
# import os

# from util.grassmann import grassmann_repr, grassmann_repr_full
# from util.glvq import *
# from lvq.prototypes import PrototypeLayer
# from lvq.measures.base_measure import GrassmannMeasureBase


# LOW_BOUND_LAMBDA = 0.001


# class Model(nn.Module):
#     def __init__(self,
#                  num_classes: int,
#                  feature_extractor: torch.nn.Module,
#                  score_fn: GrassmannMeasureBase,
#                  args: argparse.Namespace,
#                  add_on_layers: nn.Module = nn.Identity(),
#                  device='cpu'
#                  ):
#         super().__init__()

#         self._nclasses = num_classes        

#         # Set the feature extraction network
#         self.feature_extractor = feature_extractor
#         self._add_on = add_on_layers

#         # Create the prototype layers
#         self.prototype_layer = PrototypeLayer(
#             num_classes=self._nclasses,
#             score_fn=score_fn,
#             args=args,
#             device=device,
#         )

#     @property
#     def prototypes_require_grad(self) -> bool:
#         return self.prototype_layer.xprotos.requires_grad

#     @prototypes_require_grad.setter
#     def prototypes_require_grad(self, val: bool):
#         self.prototype_layer.xprotos.requires_grad = val

#     @property
#     def features_require_grad(self) -> bool:
#         return any([param.requires_grad for param in self._net.parameters()])

#     @features_require_grad.setter
#     def features_require_grad(self, val: bool):
#         for param in self._net.parameters():
#             param.requires_grad = val

#     @property
#     def add_on_layers_require_grad(self) -> bool:
#         return any([param.requires_grad for param in self._add_on.parameters()])

#     @add_on_layers_require_grad.setter
#     def add_on_layers_require_grad(self, val: bool):
#         for param in self._add_on.parameters():
#             param.requires_grad = val

#     def initialize_prototypes_from_data(self, dataloader, device):
#         """
#         Initializes the prototypes using one random sample per class from the dataloader.
#         """
#         self.eval()
#         num_classes = self._nclasses
#         selected = {}
        
#         # Loop until we have one sample per class
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             for x, y in zip(inputs, targets):
#                 y_int = int(y.item())
#                 if y_int not in selected:
#                     selected[y_int] = x.unsqueeze(0)
#                 if len(selected) == num_classes:
#                     break
#             if len(selected) == num_classes:
#                 break
        
#         if len(selected) < num_classes:
#             raise ValueError(f"Not enough classes found in the dataloader! Found {len(selected)} / {num_classes}")
        
#         # Stack and get features
#         imgs = torch.cat(list(selected.values()), dim=0).to(device)
#         with torch.no_grad():
#             features = self._add_on(self.feature_extractor(imgs))
#             subspaces = grassmann_repr(features, self.prototype_layer._subspace_dim)
        
#         # Assign these subspaces as initial prototypes
#         self.prototype_layer.xprotos.data = subspaces.clone()
#         print("✅ Prototypes initialized from data samples.")
    

#     def forward(self,
#                 inputs: torch.Tensor,
#                 **kwargs):
#         """
#         Forward pass of the model.
#         Args:
#             inputs (torch.Tensor): A batch of input data.
#         Returns:
#             tuple: Distances and Qw.
#         """
#         # Perform forward pass through the feature extractor
#         features = self.feature_extractor(inputs)
#         features = self._add_on(features)

#         # Get Grassmann representation of the features
#         subspaces = grassmann_repr(features, self.prototype_layer._subspace_dim)
#         # subspaces = grassmann_repr(features, self.prototype_layer._coef_subspace_dim * self._subspace_dim)

#         # Compute distance between subspaces and prototypes
#         distance = self.prototype_layer( # removed Qw
#             subspaces)  # SHAPE: (batch_size, num_prototypes, D: dim_of_data, d: dim_of_subspace)

#         return distance

#     def forward_partial(self,
#                         inputs: torch.Tensor):
#         # Perform forward pass through the feature extractor
#         features = self.feature_extractor(inputs)
#         features = self._add_on(features)

#         # Get Grassmann representation of the features
#         subspaces, Vh, S = grassmann_repr_full(features, self._subspace_dim)

#         output = compute_distances_on_grassmann_mdf(
#             subspaces,
#             self.prototype_layer.xprotos,
#             self._metric_type,
#             self.prototype_layer.relevances,
#         )

#         return features, subspaces, Vh, S, output

#     def save(self, directory_path: str):
#         if not os.path.isdir(directory_path):
#             os.mkdir(directory_path)

#         with open(directory_path + "/model.pth", 'wb') as f:
#             torch.save(self, f)

#     def save_state(self, directory_path: str):
#         if not os.path.isdir(directory_path):
#             os.mkdir(directory_path)

#         with open(directory_path + "/model_state.pth", 'wb') as f:
#             torch.save(self.state_dict(), f)

#     @staticmethod
#     def load(directory_path: str):
#         return torch.load(directory_path + '/model.pth', map_location=torch.device('cpu'))




# def return_model(fname):
#     with np.load(fname + '.npz', allow_pickle=True) as f:
#         xprotos, yprotos = f['xprotos'], f['yprotos']
#         lamda = f['lamda']
#         print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
#               f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

#     return xprotos, yprotos, lamda
