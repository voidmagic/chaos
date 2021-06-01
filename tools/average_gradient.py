import sys
import pathlib
import torch


folder = pathlib.Path(sys.argv[1])
files = sorted(folder.glob('[0-9]*.pt'))
average_state = {}
for file in files:
    print(file)
    state = torch.load(file)

    for lang_pair in state.keys():
        for module in state[lang_pair].keys():
            state[lang_pair][module] /= len(files)

    if not average_state:
        average_state = state
    else:
        for lang_pair in average_state.keys():
            for module in average_state[lang_pair].keys():
                average_state[lang_pair][module] += state[lang_pair][module]

torch.save(average_state, folder / 'average_grad.pt')
