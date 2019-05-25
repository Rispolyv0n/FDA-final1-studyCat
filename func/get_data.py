import json

def load(path, just_accs=False):
    with open(path) as f:
        data = json.loads(f.read())
    if just_accs:
        data = filter_accs(data)
    return data

def filter_accs(data):
    """Filters out just the records with accuracies."""
    return [x for x in data if 'accuracy' in x.keys()]
