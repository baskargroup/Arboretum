import ast

with open( r'./metadata/flowers.txt', 'r' ) as file:
    flowers_classnames = ast.literal_eval(file.read())

flowers_template = [
    lambda c: f'a photo of the {c}, a type of flower.',
    lambda c: f'a close-up photo of the {c}, a type of flower.',
    lambda c: f'a rendition of the {c}, a type of flower.',
    lambda c: f'a photo of a large {c}, a type of flower.',
    lambda c: f'itap of a {c}, a type of flower.',
]