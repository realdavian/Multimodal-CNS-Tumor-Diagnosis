import inspect

from .dataset import SyntheticDataset
from .brats import BraTSDataset
from .brats_multitask import BraTSMultitaskDataset

def create_dataset(dataset_name: str, **kwargs):
    """
    Factory function to instantiate datasets.
    
    Args:
        dataset_name (str): Identifier for the dataset (e.g., "brats_peds", "synthetic")
        **kwargs: Arguments to pass to the dataset constructor.
        
    Returns:
        torch.utils.data.Dataset
        
    Raises:
        ValueError: If the dataset name is not registered.
    """
    registry = {
        "synthetic": SyntheticDataset,
        "brats_peds": BraTSDataset,
        "brats_multitask": BraTSMultitaskDataset,
    }
    
    if dataset_name not in registry:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {list(registry.keys())}"
        )
        
    dataset_cls = registry[dataset_name]
    
    sig = inspect.signature(dataset_cls.__init__)
    valid_keys = [
        param.name 
        for param in sig.parameters.values() 
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    
    # Filter kwargs to only include those accepted by the constructor
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    return dataset_cls(**filtered_kwargs)

__all__ = [
    "SyntheticDataset",
    "BraTSDataset",
    "BraTSMultitaskDataset",
    "create_dataset",
]
