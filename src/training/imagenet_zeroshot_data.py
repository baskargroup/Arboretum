import numpy as np
import ast
from pathlib import Path

try:
    IN1K_OPENAI = Path("./metadata/imagenet1k_openclip.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_OPENAI = Path("open_clip/metadata/imagenet1k_openclip.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_classnames = ast.literal_eval( file.read( ) )

def get_imagenet_classnames():
    return imagenet_classnames

try:
    IN1K_ORDER = Path("./metadata/imagenet1k_wrongorder.txt")
    with open( IN1K_ORDER, 'r' ) as file:
        imagenet_wrongorder_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_ORDER = Path("open_clip/metadata/imagenet1k_wrongorder.txt")
    with open( IN1K_ORDER, 'r' ) as file:
        imagenet_wrongorder_classnames = ast.literal_eval( file.read( ) )

def get_imagenet_wrongorder_classnames():
    return imagenet_wrongorder_classnames

try:
    IN1K_OURS = Path("./metadata/imagenet1k_ours.txt")
    with open( IN1K_OURS, 'r' ) as file:
        imagenet_our_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_OURS = Path("open_clip/metadata/imagenet1k_ours.txt")
    with open( IN1K_OURS, 'r' ) as file:
        imagenet_our_classnames = ast.literal_eval( file.read( ) )
temp = []

for k,v in imagenet_our_classnames.items():
    temp.append("".join(v))
imagenet_our_classnames = temp

def get_imagenet_our_classnames():
    return imagenet_our_classnames

try:
    IN1K_DEF = Path("./metadata/imagenet1k_default.txt")
    with open( IN1K_DEF, 'r' ) as file:
        imagenet_def_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_DEF = Path("open_clip/metadata/imagenet1k_default.txt")
    with open( IN1K_DEF, 'r' ) as file:
        imagenet_def_classnames = ast.literal_eval( file.read( ) )
temp = []

for k,v in imagenet_def_classnames.items():
    temp.append("".join(v))
imagenet_def_classnames = temp

def get_imagenet_def_classnames():
    return imagenet_def_classnames

ir_idx = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 
125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 
231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 
296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 
362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 
463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 
637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 
889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 
983, 988]

def get_ir_idx():
    return np.array(ir_idx)

ia_idx = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 
113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 
310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 
402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 
514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 
640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 
779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 
888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 
986, 987, 988]

def get_ia_idx_zeroindexed():
    return np.arange(0, 200, dtype=int)

def get_ir_idx_zeroindexed():
    return np.arange(0, 200, dtype=int)

def get_ia_idx():
    return np.array(ia_idx)

icap_idx = [386, 928, 931, 704, 907, 291, 454, 76, 952, 788, 245, 937, 924, 8, 983, 816, 920, 379, 204, 396, 929, 619, 815, 88, 84, 217, 118, 935, 987, 642, 950, 951, 954, 557, 18, 967, 945, 6, 440, 348, 22, 571, 23, 963, 104, 958, 579, 312, 534, 620, 115, 298, 284, 552, 373, 997, 182, 422, 308, 839, 13, 489, 805, 832, 85, 695, 2, 863, 310, 565, 886, 455, 988, 347, 580, 425, 99, 424, 105, 107, 343, 658, 721, 443, 421, 679, 19, 825, 130, 309, 849, 879, 496, 971, 922, 985, 286, 625, 637, 943]
common_ia = [n for n in ia_idx if n in icap_idx]
common_ir = [n for n in ir_idx if n in icap_idx]

def get_icap_idx():
    return np.array(icap_idx)

def get_common_ia_idx():
    return np.array(common_ia)

def get_common_ir_idx():
    return np.array(common_ir)

def get_common_ia_idx_zeroindexed():
    return np.array([ia_idx.index(k) for k in common_ia])

def get_common_ir_idx_zeroindexed():
    return np.array([ir_idx.index(k) for k in common_ir])

def get_imagenet_r_classnames():
    imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[ir_idx].tolist()

def get_imagenet_r_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[ir_idx].tolist()

def get_imagenet_a_our_classnames():

    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[ia_idx].tolist()

def get_imagenet_cap_classnames():
    imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[icap_idx].tolist()

def get_imagenet_cap_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[icap_idx].tolist()

def get_imagenet_common_ia_classnames():
    imagenet_classnames_arr = np.array(imagenet_classnames)
    return imagenet_classnames_arr[common_ia].tolist()  

def get_imagenet_common_ia_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)
    return imagenet_classnames_arr[common_ia].tolist()   

def get_imagenet_common_ir_classnames():
    imagenet_classnames_arr = np.array(imagenet_classnames)
    return imagenet_classnames_arr[common_ir].tolist() 

def get_imagenet_common_ir_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)
    return imagenet_classnames_arr[common_ir].tolist() 

openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

def get_openai_imagenet_template():
    return openai_imagenet_template

try:
    IN1K_OPENAI = Path("./metadata/imagenet1k_cipher.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_cipher = ast.literal_eval( file.read( ) )
except:
    IN1K_OPENAI = Path("open_clip/metadata/imagenet1k_cipher.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_cipher = ast.literal_eval( file.read( ) )

def get_imagenet_cipher():
    return imagenet_cipher

def get_imagenet_r_cipher():
    imagenet_classnames_cipher = np.array(imagenet_cipher)
    return imagenet_classnames_cipher[ir_idx].tolist()

def get_imagenet_a_cipher():
    imagenet_classnames_cipher = np.array(imagenet_cipher)
    return imagenet_classnames_cipher[ia_idx].tolist()