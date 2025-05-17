'''
    This script contains the arguments common to all approaches.
'''
import argparse

class Arguments:

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser()
        self.__build_args(self._parser)

    def parse_args(self) -> argparse.Namespace:
        return self._parser.parse_args()

    def __build_args(self, parser) -> None:

        #
        # default arguments required in all approaches
        #

        # arguments for (down)loading the dataset from huggingface and 
        # configuring its utility
        parser.add_argument("--dataset", 
            help="The dataset to use.", 
            type=str,
            required=True)
        parser.add_argument("--token", 
            help="The token to use for downloading the dataset.", 
            type=str,
            required=True)
        parser.add_argument("--novelty", 
            help="The ratio of classes to randomly sample as novel.", 
            choices=[0.0, 0.25, 0.50, 0.75],
            type=float,
            required=False)
        parser.add_argument("--n_labels", 
            help="The list of classes to treat as novel (alternative to 'novelty', e.g., for a subsequent run in a multi-step training approach). Will always overwrite 'novelty'", 
            nargs="+",
            type=int,
            required=False)
        parser.add_argument("--error_turn", 
            help="Whether or not to use the whole error turn instead of just the error utterance.", 
            type=bool,
            required=False)
        
        # main training arguments
        parser.add_argument("--model_path", 
            help="The path to a pretrained huggingface model (the foundation model to be used).", 
            type=str,
            required=False)               
        parser.add_argument("--pretrained", 
            help="The path to a model pretrained using this framework.", 
            type=str,
            required=False)        
        parser.add_argument("--batch_size", 
            help="The number of samples per batch.", 
            type=int,
            required=True)           
        parser.add_argument("--epochs", 
            help="The number of epochs for the main training.", 
            type=int,
            required=False)
        parser.add_argument("--device", 
            help="The device to use for training ('cuda' or 'cpu').", 
            type=str,
            choices=["cuda", "cpu"],
            default="cuda",
            required=True)
        parser.add_argument("--save_dir", 
            help="The directory where to save the trained model.", 
            type=str,
            required=False)        
        
        # meta arguments; configuration for MLFlow and additional output after 
        # evaluation
        parser.add_argument("--experiment", 
            help="The name of the experiment (for MLFlow.)", 
            type=str,
            required=True)
        parser.add_argument("--visualize", 
            help="Whether or not to visualize the after evaluation.", 
            type=bool,
            required=False)   
        
    @property
    def parser(self):
        return self._parser