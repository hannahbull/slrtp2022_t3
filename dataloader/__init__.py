from .spotting_dataloader import SpottingDataset

dataset_dict = {
    'spotting': SpottingDataset,
}

__all__ = ['dataset_dict']