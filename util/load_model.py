import pickle
import torch
import torch.nn as nn
import argparse
from lvq.model import GrassmannLVQModel
from util.net import get_network
from lvq.measures.angle_measure import AngleMeasure



def load_grassmannlvq_model(args_path: str = None,
                            args: argparse.Namespace = None,
                            checkpoint_path: str = None,
                            device: str = "cpu",
                            trainloader: torch.utils.data.DataLoader = None,
                            init_from_data: bool = False) -> GrassmannLVQModel:
    """
    Load GrassmannLVQModel from saved args and checkpoint.

    args_path: path to args.pkl containing argparse.Namespace
    checkpoint_path: e.g., ".../latest" or ".../epoch_20"
    device: "cpu" or "cuda"
    trainloader: dataloader needed for prototype initialization
    """

    # 1. Load saved args
    if args is None and args_path is not None:
        with open(args_path, "rb") as f:
            args = pickle.load(f)
    elif args is None and args_path is None:
        raise ValueError("Either args or args_path must be provided.")

    # 2. Rebuild sub-networks (must match training code)
    features_net, add_on_layers = get_network(args.__dict__.get('num_channels', 3), args)
    
    score_fn = AngleMeasure(beta=args.beta)    

    # 3. Load saved weights
    if checkpoint_path is not None:
        model = GrassmannLVQModel.load_state(
            checkpoint_path,
            GrassmannLVQModel,
            feature_extractor=features_net,
            score_fn=score_fn,
            args=args,
            add_on_layers=add_on_layers if args.num_features > 0 else nn.Identity(),
            device=device,
            # init_from_data=False,
            # dataloader=trainloader,
        )
    else:
        # 3. Rebuild the model architecture
        model = GrassmannLVQModel(
            feature_extractor=features_net,
            score_fn=score_fn,
            args=args,
            add_on_layers=add_on_layers if args.num_features > 0 else nn.Identity(),
            device=device,
            init_from_data=init_from_data,
            dataloader=trainloader,
        )


    model.to(device)

    if checkpoint_path is not None: 
        model.eval()

    return model, args
