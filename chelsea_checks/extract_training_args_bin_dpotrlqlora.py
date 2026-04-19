import torch

#load the training_args.bin file found in the dpo model using dpotrainer from trl to extract all hyperparameters used
training_args = torch.load("training_args.bin")

print(training_args)