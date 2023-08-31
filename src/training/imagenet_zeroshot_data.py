import numpy as np
import ast
import json
from pathlib import Path

try:
    IN21K_TO_1K = Path("./metadata/imagenet21k_to_1k_index.txt")
    with open( IN21K_TO_1K, 'r' ) as file:
        in21k_to_1k = ast.literal_eval( file.read( ) )
except:
    IN21K_TO_1K = Path("/scratch/bf996/vlhub/metadata/imagenet21k_to_1k_index.txt")
    with open( IN21K_TO_1K, 'r' ) as file:
        in21k_to_1k = ast.literal_eval( file.read( ) )

def get_in21k_to_1k():
    return in21k_to_1k

import json

try:
    IN1K_WNID_IDX = Path("./metadata/imagenet1k_wnid_to_index.json")
    with open( IN1K_WNID_IDX, 'r' ) as file:
        in1k_wnid_idx = json.load( file )
except:
    IN1K_WNID_IDX = Path("/scratch/bf996/vlhub/metadata/imagenet1k_wnid_to_index.json")
    with open( IN1K_WNID_IDX, 'r' ) as file:
        in1k_wnid_idx = json.load( file )

def get_in1k_wnid_to_idx():
    return in1k_wnid_idx

try:
    IN21K_WNID_IDX = Path("./metadata/imagenet21k_wnid_to_idx.json")
    with open( IN21K_WNID_IDX, 'r' ) as file:
        in21k_wnid_idx = json.load( file )
except:
    IN21K_WNID_IDX = Path("/scratch/bf996/vlhub/metadata/imagenet21k_wnid_to_idx.json")
    with open( IN21K_WNID_IDX, 'r' ) as file:
        in21k_wnid_idx = json.load( file )

def get_in21k_wnid_to_idx():
    return in21k_wnid_idx

try:
    IN1K_OPENAI = Path("./metadata/imagenet1k_openclip.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_OPENAI = Path("/scratch/bf996/vlhub/metadata/imagenet1k_openclip.txt")
    with open( IN1K_OPENAI, 'r' ) as file:
        imagenet_classnames = ast.literal_eval( file.read( ) )

try:
    IN1K_NO_OVERLAP = Path("./metadata/imagenet_no_overlap_classnames.json")
    with open( IN1K_NO_OVERLAP, 'r' ) as file:
        imagenet_no_overlap_classnames = json.load(file)
except:
    IN1K_NO_OVERLAP = Path("/scratch/bf996/vlhub/metadata/imagenet_no_overlap_classnames.json")
    with open( IN1K_NO_OVERLAP, 'r' ) as file:
        imagenet_no_overlap_classnames = json.load(file)

try:
    IN1K_NO_OVERLAP_SHORT = Path("./metadata/imagenet_no_overlap_short_classnames.json")
    with open( IN1K_NO_OVERLAP_SHORT, 'r' ) as file:
        imagenet_no_overlap_s_classnames = json.load(file)
except:
    IN1K_NO_OVERLAP_SHORT = Path("/scratch/bf996/vlhub/metadata/imagenet_no_overlap_short_classnames.json")
    with open( IN1K_NO_OVERLAP_SHORT, 'r' ) as file:
        imagenet_no_overlap_s_classnames = json.load(file)

try:
    IN1K_NO_PERM_SHORT = Path("./metadata/imagenet_no_perm_classnames.json")
    with open( IN1K_NO_PERM_SHORT, 'r' ) as file:
        imagenet_no_perm_classnames = json.load(file)
except:
    IN1K_NO_PERM_SHORT = Path("/scratch/bf996/vlhub/metadata/imagenet_no_perm_classnames.json")
    with open( IN1K_NO_PERM_SHORT, 'r' ) as file:
        imagenet_no_perm_classnames = json.load(file)

def get_imagenet_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        return imagenet_no_overlap_classnames
    elif short_no_overlap:
        return imagenet_no_overlap_s_classnames
    elif no_perm:
        return imagenet_no_perm_classnames
    return imagenet_classnames

try:
    IN1K_ORDER = Path("./metadata/imagenet1k_wrongorder.txt")
    with open( IN1K_ORDER, 'r' ) as file:
        imagenet_wrongorder_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_ORDER = Path("/scratch/bf996/vlhub/metadata/imagenet1k_wrongorder.txt")
    with open( IN1K_ORDER, 'r' ) as file:
        imagenet_wrongorder_classnames = ast.literal_eval( file.read( ) )

def get_imagenet_wrongorder_classnames():
    return imagenet_wrongorder_classnames

try:
    IN1K_OURS = Path("./metadata/imagenet1k_ours.txt")
    with open( IN1K_OURS, 'r' ) as file:
        imagenet_our_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_OURS = Path("/scratch/bf996/vlhub/metadata/imagenet1k_ours.txt")
    with open( IN1K_OURS, 'r' ) as file:
        imagenet_our_classnames = ast.literal_eval( file.read( ) )
temp = []

for k,v in imagenet_our_classnames.items():
    temp.append("".join(v))
imagenet_our_classnames = temp

ia_idx = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 
113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 
310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 
402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 
514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 
640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 
779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 
888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 
986, 987, 988]


ir_idx = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 
125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 
231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 
296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 
362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 
463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 
637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 
889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 
983, 988]

obj_idx_dict = {'alarm_clock': [409, 530], 'backpack': [414], 'banana': [954], 'band_aid': [419], 'basket': [790], 'full_sized_towel': [434], 'beer_bottle': [440], 'bench': [703], 'bicycle': [671, 444], 'binder_closed': [446], 'bottle_cap': [455], 'bread_loaf': [930], 'broom': [462], 'bucket': [463], 'butchers_knife': [499], 'can_opener': [473], 'candle': [470], 'cellphone': [487], 'chair': [423, 559, 765], 'clothes_hamper': [588], 'coffee_french_press': [550], 'combination_lock': [507], 'computer_mouse': [673], 'desk_lamp': [846], 'hand_towel_or_rag': [533], 'doormat': [539], 'dress_shoe_men': [630], 'drill': [740], 'drinking_cup': [968], 'drying_rack_for_dishes': [729], 'envelope': [549], 'fan': [545], 'frying_pan': [567], 'dress': [578], 'hair_dryer': [589], 'hammer': [587], 'helmet': [560, 518], 'iron_for_clothes': [606], 'jeans': [608], 'keyboard': [508], 'ladle': [618], 'lampshade': [619], 'laptop_open': [620], 'lemon': [951], 'letter_opener': [623], 'lighter': [626], 'lipstick': [629], 'match': [644], 'measuring_cup': [647], 'microwave': [651], 'mixing_salad_bowl': [659], 'monitor': [664], 'mug': [504], 'nail_fastener': [677], 'necklace': [679], 'orange': [950], 'padlock': [695], 'paintbrush': [696], 'paper_towel': [700], 'pen': [418, 749, 563], 'pill_bottle': [720], 'pillow': [721], 'pitcher': [725], 'plastic_bag': [728], 'plate': [923], 'plunger': [731], 'pop_can': [737], 'portable_heater': [811], 'printer': [742], 'remote_control': [761], 'ruler': [769], 'running_shoe': [770], 'safety_pin': [772], 'salt_shaker': [773], 'sandal': [774], 'screw': [783], 'shovel': [792], 'skirt': [601, 655, 689], 'sleeping_bag': [797], 'soap_dispenser': [804], 'sock': [806], 'soup_bowl': [809], 'spatula': [813], 'speaker': [632], 'still_camera': [732, 759], 'strainer': [828], 'stuffed_animal': [850], 'suit_jacket': [834], 'sunglasses': [837], 'sweater': [841], 'swimming_trunks': [842], 't-shirt': [610], 'tv': [851], 'teapot': [849], 'tennis_racket': [752], 'tie': [457, 906], 'toaster': [859], 'toilet_paper_roll': [999], 'trash_bin': [412], 'tray': [868], 'umbrella': [879], 'vacuum_cleaner': [882], 'vase': [883], 'wallet': [893], 'watch': [531], 'water_bottle': [898], 'weight_exercise': [543], 'weight_scale': [778], 'wheel': [479, 694], 'whistle': [902], 'wine_bottle': [907], 'winter_glove': [658], 'wok': [909]}

obj_idx = [409, 530, 414, 954, 419, 790, 434, 440, 703, 671, 444, 446, 455, 930, 462, 463, 499, 473, 470, 487, 423, 559, 765, 588, 550, 507, 673, 846, 533, 539, 630, 740, 968, 729, 549, 545, 567, 578, 589, 587, 560, 518, 606, 608, 508, 618, 619, 620, 951, 623, 626, 629, 644, 647, 651, 659, 664, 504, 677, 679, 950, 695, 696, 700, 418, 749, 563, 720, 721, 725, 728, 923, 731, 737, 811, 742, 761, 769, 770, 772, 773, 774, 783, 792, 601, 655, 689, 797, 804, 806, 809, 813, 632, 732, 759, 828, 850, 834, 837, 841, 842, 610, 851, 849, 752, 457, 906, 859, 999, 412, 868, 879, 882, 883, 893, 531, 898, 543, 778, 479, 694, 902, 907, 658, 909]

def get_imagenet_our_classnames():
    return imagenet_our_classnames

try:
    IN1K_DEF = Path("./metadata/imagenet1k_default.txt")
    with open( IN1K_DEF, 'r' ) as file:
        imagenet_def_classnames = ast.literal_eval( file.read( ) )
except:
    IN1K_DEF = Path("/scratch/bf996/vlhub/metadata/imagenet1k_default.txt")
    with open( IN1K_DEF, 'r' ) as file:
        imagenet_def_classnames = ast.literal_eval( file.read( ) )
temp = []

for k,v in imagenet_def_classnames.items():
    temp.append("".join(v))
imagenet_def_classnames = temp

def get_imagenet_def_classnames():
    return imagenet_def_classnames

def hex_to_decimal(hex_string):
    return int(hex_string, 16)

start_d = hex_to_decimal('4E00')
end_d = hex_to_decimal('9FFF')

#ideograms = [''.join(chr(i+j) for j in range(2)) for i in range(start_d, end_d, 2)]
ideograms = [chr(i) for i in range(start_d, end_d)]

def get_ideogram_dict():
    return {i: ideograms[i] for i in range(len(ideograms))}

def get_imagenet_ideo_classnames():
    return [ideograms[i] for i in range(1000)]

def get_ir_idx():
    return np.array(ir_idx)

def get_ia_idx_zeroindexed():
    return np.arange(0, 200, dtype=int)

def get_ir_idx_zeroindexed():
    return np.arange(0, 200, dtype=int)

def get_ia_idx():
    return np.array(ia_idx)

def get_obj_index():
    return np.array(obj_idx)

def get_obj_index_zeroindexed():
    return np.arange(0, len(obj_idx), dtype=int)

#./metadata/in100_rand_idx_{01 .. 09}.txt for a random 100-class index that is NOT in100 overlapping
#./metadata/in100_true_idx.txt for the in100 index
try:
    ICAP_F = Path("./metadata/in100_true_idx.txt")
    with open( ICAP_F, 'r' ) as file:
        icap_idx = ast.literal_eval( file.read( ) )
except:
    ICAP_F = Path("/scratch/bf996/vlhub/metadata/in100_true_idx.txt")
    with open( ICAP_F, 'r' ) as file:
        icap_idx = ast.literal_eval( file.read( ) )

try:
    IN100_DOGS = Path("./metadata/in100_dogs.txt")
    with open( IN100_DOGS, 'r' ) as file:
        dogs_idx = ast.literal_eval( file.read( ) )
except:
    IN100_DOGS = Path("/scratch/bf996/vlhub/metadata/in100_dogs.txt")
    with open( IN100_DOGS, 'r' ) as file:
        dogs_idx = ast.literal_eval( file.read( ) )
    
try:
    IN100_RANDOM_01 = Path("./metadata/in100_rand_idx_01.txt")
    with open ( IN100_RANDOM_01, 'r' ) as file:
        in100_random_01_idx = ast.literal_eval( file.read( ) )
except:
    IN100_RANDOM_01 = Path("/scratch/bf996/vlhub/metadata/in100_rand_idx_01.txt")
    with open ( IN100_RANDOM_01, 'r' ) as file:
        in100_random_01_idx = ast.literal_eval( file.read( ) )

try:
    IN100_RANDOM_02 = Path("./metadata/in100_rand_idx_02.txt")
    with open ( IN100_RANDOM_02, 'r' ) as file:
        in100_random_02_idx = ast.literal_eval( file.read( ) )
except:
    IN100_RANDOM_02 = Path("/scratch/bf996/vlhub/metadata/in100_rand_idx_02.txt")
    with open ( IN100_RANDOM_02, 'r' ) as file:
        in100_random_02_idx = ast.literal_eval( file.read( ) )

try:
    IN100_RANDOM_03 = Path("./metadata/in100_rand_idx_03.txt")
    with open ( IN100_RANDOM_03, 'r' ) as file:
        in100_random_03_idx = ast.literal_eval( file.read( ) )
except:
    IN100_RANDOM_03 = Path("/scratch/bf996/vlhub/metadata/in100_rand_idx_03.txt")
    with open ( IN100_RANDOM_03, 'r' ) as file:
        in100_random_03_idx = ast.literal_eval( file.read( ) )

def get_icap_idx(target):
    global icap_idx
    global dogs_idx
    global in100_random_01_idx
    global in100_random_02_idx
    global in100_random_03_idx
    icap_idx_rem = None
    if target == "in100":
        icap_idx_rem = np.array(icap_idx)
    elif target == "in100_dogs":
        icap_idx_rem = np.array(dogs_idx)
    elif target == "in100_random_01":
        icap_idx_rem = np.array(in100_random_01_idx)
    elif target == "in100_random_02":
        icap_idx_rem = np.array(in100_random_02_idx)
    elif target == "in100_random_03":
        icap_idx_rem = np.array(in100_random_03_idx)
    else:
        print("no match found")
    icap_idx = icap_idx_rem
    return icap_idx_rem

common_ia = [n for n in ia_idx if n in icap_idx]
common_ir = [n for n in ir_idx if n in icap_idx]
common_obj = [n for n in obj_idx if n in icap_idx]

def get_common_ia_idx():
    global common_ia
    common_ia = np.array([n for n in ia_idx if n in icap_idx])
    return common_ia

def get_common_ir_idx():
    global common_ir
    common_ir_loc = np.array([n for n in ir_idx if n in icap_idx])
    common_ir = common_ir_loc
    return common_ir_loc

def get_common_obj_idx():
    global common_obj
    common_obj_loc = np.array([n for n in obj_idx if n in icap_idx])
    common_obj = common_obj_loc
    return common_obj_loc

def get_common_ia_idx_zeroindexed():
    global common_ia
    return np.array([ia_idx.index(k) for k in common_ia])

def get_common_ir_idx_zeroindexed():
    global common_ir
    return np.array([ir_idx.index(k) for k in common_ir])

def get_common_obj_idx_zeroindexed():
    global common_obj
    return np.array([obj_idx.index(k) for k in common_obj])

def get_objectnet_classnames():
    imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[obj_idx].tolist()

def get_imagenet_r_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[ir_idx].tolist()

def get_imagenet_r_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[ir_idx].tolist()

def get_imagenet_r_ideo_classnames():
    return [ideograms[i] for i in ir_idx]

def get_imagenet_a_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[ia_idx].tolist()

def get_imagenet_a_our_classnames():

    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[ia_idx].tolist()

def get_imagenet_a_ideo_classnames():
    return [ideograms[i] for i in ia_idx]

def get_imagenet_cap_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)

    return imagenet_classnames_arr[icap_idx].tolist()

def get_imagenet_cap_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)

    return imagenet_classnames_arr[icap_idx].tolist()

def get_imagenet_cap_ideo_classnames():
    return [ideograms[i] for i in icap_idx]

def get_imagenet_common_ia_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)
    return imagenet_classnames_arr[common_ia].tolist()  

def get_imagenet_common_ia_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)
    return imagenet_classnames_arr[common_ia].tolist()   

def get_imagenet_common_ia_ideo_classnames():
    return [ideograms[i] for i in common_ia]

def get_imagenet_common_ir_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)
    return imagenet_classnames_arr[common_ir].tolist() 

def get_imagenet_common_ir_our_classnames():
    imagenet_classnames_arr = np.array(imagenet_our_classnames)
    return imagenet_classnames_arr[common_ir].tolist()

def get_imagenet_common_ir_ideo_classnames():
    return [ideograms[i] for i in common_ir]

def get_imagenet_common_obj_classnames(no_overlap=False, short_no_overlap=False, no_perm=False):
    if no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_classnames)
    elif short_no_overlap:
        imagenet_classnames_arr = np.array(imagenet_no_overlap_s_classnames)
    elif no_perm:
        imagenet_classnames_arr = np.array(imagenet_no_perm_classnames)
    else:
        imagenet_classnames_arr = np.array(imagenet_classnames)
    return imagenet_classnames_arr[common_obj].tolist()

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
    IN1K_OPENAI = Path("/scratch/bf996/vlhub/metadata/imagenet1k_cipher.txt")
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

def get_imagenet_synonym_classnames(seed=0):
    import random
    random.seed(seed)
    s = random.randint(0, 9)
    import json
    with open('/scratch/bf996/vlhub/metadata/in1k_gpt_synonyms_indexed.json','r') as f:
        gpt_syns = json.load(f)
        counter = 0
        for k, v in gpt_syns.items():
            counter += 1
            gpt_syns[k] = str(v[(s + counter) % 10])
        gpt_syns = np.array(list(gpt_syns.values()))
        gpt_r_classnames = gpt_syns[ir_idx].tolist()
        gpt_a_classnames = gpt_syns[ia_idx].tolist()
        gpt_cap_classnames = gpt_syns[icap_idx].tolist()
        gpt_common_ia_classnames = gpt_syns[common_ia].tolist()
        gpt_common_ir_classnames = gpt_syns[common_ir].tolist()
        return gpt_syns, gpt_r_classnames, gpt_a_classnames, gpt_cap_classnames, gpt_common_ir_classnames, gpt_common_ia_classnames
    
def get_all_imagenet_default_classnames(first_only=False):
    imagenet_def_classnames = get_imagenet_def_classnames()
    if first_only:
        imagenet_def_classnames = [i.split(', ')[0].strip() for i in imagenet_def_classnames]
    imagenet_def_classnames = np.array(imagenet_def_classnames)
    return imagenet_def_classnames, imagenet_def_classnames[ir_idx].tolist(),imagenet_def_classnames[ia_idx].tolist(), imagenet_def_classnames[icap_idx].tolist(), imagenet_def_classnames[common_ir].tolist(), imagenet_def_classnames[common_ia].tolist()