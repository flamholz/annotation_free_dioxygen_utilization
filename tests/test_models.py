
# TODO: Test to make sure that the evaluated accuracies match the "best" validation accuracy of the model. 
# TODO: Should I keep tests to trained models? That would be easiest. 
# TODO: Probably should do something to make sure the weights are different than initialized weights.

from aerobot.models import evaluate, GeneralClassifier 
from aerobot.io import DATA_PATH, MODELS_PATH
from dataset import dataset_load_training_testing_validation