import os
import numpy as np
import  shutil 
import pickle
import cv2
import random
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Label =['Aberts_Towhee', 'Acorn_Woodpecker', 'Allens_Hummingbird_Adult_Male', 'Allens_Hummingbird_Female,_immature', 'American_Avocet', 'American_Black_Duck', 'American_Coot', 'American_Crow', 'American_Dipper', 'American_Goldfinch_Breeding_Male', 'American_Goldfinch_Female_OR_Nonbreeding_Male', 'American_Kestrel_Adult_male', 'American_Kestrel_Female,_immature', 'American_Oystercatcher', 'American_Pipit', 'American_Redstart_Adult_Male', 'American_Redstart_Female_OR_juvenile', 'American_Robin_Adult', 'American_Robin_Juvenile', 'American_Tree_Sparrow', 'American_White_Pelican', 'American_Wigeon_Breeding_male', 'American_Wigeon_Female_OR_Eclipse_male', 'American_Woodcock', 'Anhinga', 'Annas_Hummingbird_Adult_Male', 'Annas_Hummingbird_Female,_immature', 'Ash_throated_Flycatcher', 'Bald_Eagle_Adult,_subadult', 'Bald_Eagle_Immature,_juvenile', 'Baltimore_Oriole_Adult_male', 'Baltimore_Oriole_Female_OR_Immature_male', 'Band_tailed_Pigeon', 'Bank_Swallow', 'Barn_Owl', 'Barn_Swallow', 'Barred_Owl', 'Barrows_Goldeneye_Breeding_male', 'Barrows_Goldeneye_Female_OR_Eclipse_male', 'Bay_breasted_Warbler_Breeding_male', 'Bay_breasted_Warbler_Female,_Nonbreeding_male,_Immature', 'Bells_Vireo', 'Belted_Kingfisher', 'Bewicks_Wren', 'Blackburnian_Warbler', 'Blackpoll_Warbler_Breeding_male', 'Blackpoll_Warbler_Female_OR_juvenile', 'Black_and_white_Warbler', 'Black_bellied_Plover_Breeding', 'Black_bellied_Plover_Nonbreeding_OR_juvenile', 'Black_bellied_Whistling_Duck', 'Black_billed_Cuckoo', 'Black_billed_Magpie', 'Black_capped_Chickadee', 'Black_chinned_Hummingbird_Adult_Male', 'Black_chinned_Hummingbird_Female,_immature', 'Black_crested_Titmouse', 'Black_crowned_Night_Heron_Adult', 'Black_crowned_Night_Heron_Immature', 'Black_Guillemot_Breeding', 'Black_Guillemot_Nonbreeding,_juvenile', 'Black_headed_Grosbeak_Adult_Male', 'Black_headed_Grosbeak_Female_OR_immature_male', 'Black_legged_Kittiwake_Adult', 'Black_legged_Kittiwake_Immature', 'Black_necked_Stilt', 'Black_Oystercatcher', 'Black_Phoebe', 'Black_Rosy_Finch', 'Black_Scoter_Female_OR_juvenile', 'Black_Scoter_Male', 'Black_Skimmer', 'Black_tailed_Gnatcatcher', 'Black_Tern', 'Black_throated_Blue_Warbler_Adult_Male', 'Black_throated_Blue_Warbler_Female_OR_Immature_male', 'Black_throated_Gray_Warbler', 'Black_throated_Green_Warbler', 'Black_Turnstone', 'Black_Vulture', 'Blue_gray_Gnatcatcher', 'Blue_Grosbeak_Adult_Male', 'Blue_Grosbeak_Female_OR_juvenile', 'Blue_headed_Vireo', 'Blue_Jay', 'Blue_winged_Teal__Female_OR_juvenile', 'Blue_winged_Teal__Male', 'Blue_winged_Warbler', 'Boat_tailed_Grackle', 'Bobolink_Breeding_male', 'Bobolink_Female_OR_juvenile_OR_nonbreeding_male', 'Bohemian_Waxwing', 'Bonapartes_Gull', 'Boreal_Chickadee', 'Brandts_Cormorant', 'Brant', 'Brewers_Blackbird_Female_OR_Juvenile', 'Brewers_Blackbird_Male', 'Brewers_Sparrow', 'Bridled_Titmouse', 'Broad_billed_Hummingbird_Adult_Male', 'Broad_billed_Hummingbird_Female,_immature', 'Broad_tailed_Hummingbird_Adult_Male', 'Broad_tailed_Hummingbird_Female,_immature', 'Broad_winged_Hawk_Adult', 'Broad_winged_Hawk_Immature', 'Bronzed_Cowbird', 'Brown_capped_Rosy_Finch', 'Brown_Creeper', 'Brown_headed_Cowbird_Female_OR_Juvenile', 'Brown_headed_Cowbird_Male', 'Brown_headed_Nuthatch', 'Brown_Pelican', 'Brown_Thrasher', 'Bufflehead_Breeding_male', 'Bufflehead_Female_OR_immature_male', 'Bullocks_Oriole_Adult_male', 'Bullocks_Oriole_Female_OR_Immature_male', 'Burrowing_Owl', 'Bushtit', 'Cackling_Goose', 'Cactus_Wren', 'California_Gull_Adult', 'California_Gull_Immature', 'California_Quail_Female_OR_juvenile', 'California_Quail_Male', 'California_Thrasher', 'California_Towhee', 'Calliope_Hummingbird_Adult_Male', 'Calliope_Hummingbird_Female,_immature', 'Canada_Goose', 'Canada_Warbler', 'Canvasback_Breeding_male', 'Canvasback_Female_OR_Eclipse_male', 'Canyon_Towhee', 'Canyon_Wren', 'Cape_May_Warbler', 'Carolina_Chickadee', 'Carolina_Wren', 'Caspian_Tern', 'Cassins_Finch_Adult_Male', 'Cassins_Finch_Female_OR_immature', 'Cassins_Kingbird', 'Cassins_Vireo', 'Cattle_Egret', 'Cave_Swallow', 'Cedar_Waxwing', 'Chestnut_backed_Chickadee', 'Chestnut_sided_Warbler_Breeding_male', 'Chestnut_sided_Warbler_Female_OR_immature_male', 'Chihuahuan_Raven', 'Chimney_Swift', 'Chipping_Sparrow_Breeding', 'Chipping_Sparrow_Immature_OR_nonbreeding_adult', 'Cinnamon_Teal_Female_OR_juvenile', 'Cinnamon_Teal_Male', 'Clarks_Grebe', 'Clarks_Nutcracker', 'Clay_colored_Sparrow', 'Cliff_Swallow', 'Common_Eider_Adult_male', 'Common_Eider_Female_OR_juvenile', 'Common_Eider_Immature_OR_Eclipse_male', 'Common_Gallinule_Adult', 'Common_Gallinule_Immature', 'Common_Goldeneye_Breeding_male', 'Common_Goldeneye_Female_OR_Eclipse_male', 'Common_Grackle', 'Common_Ground_Dove', 'Common_Loon_Breeding', 'Common_Loon_Nonbreeding_OR_juvenile', 'Common_Merganser_Breeding_male', 'Common_Merganser_Female_OR_immature_male', 'Common_Nighthawk', 'Common_Raven', 'Common_Redpoll', 'Common_Tern', 'Common_Yellowthroat_Adult_Male', 'Common_Yellowthroat_Female_OR_immature_male', 'Coopers_Hawk_Adult', 'Coopers_Hawk_Immature', 'Cordilleran_Flycatcher', 'Costas_Hummingbird_Adult_Male', 'Costas_Hummingbird_Female,_immature', 'Crested_Caracara', 'Curve_billed_Thrasher', 'Dark_eyed_Junco_Oregon', 'Dark_eyed_Junco_Pink_sided', 'Dark_eyed_Junco_Red_backed_OR_Gray_headed', 'Dark_eyed_Junco_Slate_colored', 'Dark_eyed_Junco_White_winged', 'Dickcissel', 'Double_crested_Cormorant_Adult', 'Double_crested_Cormorant_Immature', 'Downy_Woodpecker', 'Dunlin_Breeding', 'Dunlin_Nonbreeding_OR_juvenile', 'Eared_Grebe_Breeding', 'Eared_Grebe_Nonbreeding_OR_juvenile', 'Eastern_Bluebird', 'Eastern_Kingbird', 'Eastern_Meadowlark', 'Eastern_Phoebe', 'Eastern_Screech_Owl', 'Eastern_Towhee', 'Eastern_Wood_Pewee', 'Eurasian_Collared_Dove', 'European_Starling_Breeding_Adult', 'European_Starling_Juvenile', 'European_Starling_Nonbreeding_Adult', 'Evening_Grosbeak_Adult_Male', 'Evening_Grosbeak_Female_OR_Juvenile', 'Field_Sparrow', 'Fish_Crow', 'Florida_Scrub_Jay', 'Forsters_Tern', 'Fox_Sparrow_Red', 'Fox_Sparrow_Sooty', 'Fox_Sparrow_Thick_billed_OR_Slate_colored', 'Gadwall_Breeding_male', 'Gadwall_Female_OR_Eclipse_male', 'Gambels_Quail_Female_OR_juvenile', 'Gambels_Quail_Male', 'Gila_Woodpecker', 'Glaucous_winged_Gull_Adult', 'Glaucous_winged_Gull_Immature', 'Glossy_Ibis', 'Golden_crowned_Kinglet', 'Golden_crowned_Sparrow_Adult', 'Golden_crowned_Sparrow_Immature', 'Golden_Eagle_Adult', 'Golden_Eagle_Immature', 'Golden_fronted_Woodpecker', 'Gray_Catbird', 'Gray_crowned_Rosy_Finch', 'Gray_Jay', 'Greater_Roadrunner', 'Greater_Scaup_Breeding_male', 'Greater_Scaup_Female_OR_Eclipse_male', 'Greater_White_fronted_Goose', 'Greater_Yellowlegs', 'Great_Black_backed_Gull_Adult', 'Great_Black_backed_Gull_Immature', 'Great_Blue_Heron', 'Great_Cormorant_Adult', 'Great_Cormorant_Immature', 'Great_Crested_Flycatcher', 'Great_Egret', 'Great_Horned_Owl', 'Great_tailed_Grackle', 'Green_Heron', 'Green_tailed_Towhee', 'Green_winged_Teal_Male', 'Green_winged_Teal__Female_OR_juvenile', 'Hairy_Woodpecker', 'Harlequin_Duck_Female_OR_juvenile', 'Harlequin_Duck_Male', 'Harriss_Hawk', 'Harriss_Sparrow_Adult', 'Harriss_Sparrow_Immature', 'Heermanns_Gull_Adult', 'Heermanns_Gull_Immature', 'Hermit_Thrush', 'Hermit_Warbler', 'Herring_Gull_Adult', 'Herring_Gull_Immature', 'Hoary_Redpoll', 'Hooded_Merganser_Breeding_male', 'Hooded_Merganser_Female_OR_immature_male', 'Hooded_Oriole_Adult_male', 'Hooded_Oriole_Female_OR_Immature_male', 'Hooded_Warbler', 'Horned_Grebe_Breeding', 'Horned_Grebe_Nonbreeding_OR_juvenile', 'Horned_Lark', 'House_Finch_Adult_Male', 'House_Finch_Female_OR_immature', 'House_Sparrow_Female_OR_Juvenile', 'House_Sparrow_Male', 'House_Wren', 'Huttons_Vireo', 'Inca_Dove', 'Indigo_Bunting_Adult_Male', 'Indigo_Bunting_Female_OR_juvenile', 'Juniper_Titmouse', 'Killdeer', 'Ladder_backed_Woodpecker', 'Lark_Bunting_Breeding_male', 'Lark_Bunting_Female_OR_Nonbreeding_male', 'Lark_Sparrow', 'Laughing_Gull_Breeding', 'Laughing_Gull_Nonbreeding_OR_Immature', 'Lazuli_Bunting_Adult_Male', 'Lazuli_Bunting_Female_OR_juvenile', 'Least_Flycatcher', 'Least_Sandpiper', 'Lesser_Goldfinch_Adult_Male', 'Lesser_Goldfinch_Female_OR_juvenile', 'Lesser_Scaup_Breeding_male', 'Lesser_Scaup_Female_OR_Eclipse_male', 'Lesser_Yellowlegs', 'Lincolns_Sparrow', 'Little_Blue_Heron_Adult', 'Little_Blue_Heron_Immature', 'Loggerhead_Shrike', 'Long_billed_Curlew', 'Long_tailed_Duck_Female_OR_juvenile', 'Long_tailed_Duck_Summer_male', 'Long_tailed_Duck_Winter_male', 'Louisiana_Waterthrush', 'MacGillivrays_Warbler', 'Magnolia_Warbler_Breeding_male', 'Magnolia_Warbler_Female_OR_immature_male', 'Mallard_Breeding_male', 'Mallard_Female_OR_Eclipse_male', 'Marbled_Godwit', 'Marsh_Wren', 'Merlin', 'Mew_Gull', 'Mexican_Jay', 'Mississippi_Kite', 'Monk_Parakeet', 'Mottled_Duck', 'Mountain_Bluebird', 'Mountain_Chickadee', 'Mourning_Dove', 'Mourning_Warbler', 'Mute_Swan', 'Nashville_Warbler', 'Neotropic_Cormorant', 'Northern_Bobwhite', 'Northern_Cardinal_Adult_Male', 'Northern_Cardinal_Female_OR_Juvenile', 'Northern_Flicker_Red_shafted', 'Northern_Flicker_Yellow_shafted', 'Northern_Gannet_Adult,_Subadult', 'Northern_Gannet_Immature_OR_Juvenile', 'Northern_Harrier_Adult_male', 'Northern_Harrier_Female,_immature', 'Northern_Mockingbird', 'Northern_Parula', 'Northern_Pintail_Breeding_male', 'Northern_Pintail_Female_OR_Eclipse_male', 'Northern_Pygmy_Owl', 'Northern_Rough_winged_Swallow', 'Northern_Saw_whet_Owl', 'Northern_Shoveler_Breeding_male', 'Northern_Shoveler_Female_OR_Eclipse_male', 'Northern_Shrike', 'Northern_Waterthrush', 'Northwestern_Crow', 'Nuttalls_Woodpecker', 'Oak_Titmouse', 'Orange_crowned_Warbler', 'Orchard_Oriole_Adult_Male', 'Orchard_Oriole_Female_OR_Juvenile', 'Orchard_Oriole_Immature_Male', 'Osprey', 'Ovenbird', 'Pacific_Loon_Breeding', 'Pacific_Loon_Nonbreeding_OR_juvenile', 'Pacific_slope_Flycatcher', 'Pacific_Wren', 'Painted_Bunting_Adult_Male', 'Painted_Bunting_Female_OR_juvenile', 'Palm_Warbler', 'Pelagic_Cormorant', 'Peregrine_Falcon_Adult', 'Peregrine_Falcon_Immature', 'Phainopepla_Female_OR_juvenile', 'Phainopepla_Male', 'Pied_billed_Grebe', 'Pigeon_Guillemot_Breeding', 'Pigeon_Guillemot_Nonbreeding,_juvenile', 'Pileated_Woodpecker', 'Pine_Grosbeak_Adult_Male', 'Pine_Grosbeak_Female_OR_juvenile', 'Pine_Siskin', 'Pine_Warbler', 'Plumbeous_Vireo', 'Prairie_Falcon', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Purple_Finch_Adult_Male', 'Purple_Finch_Female_OR_immature', 'Purple_Gallinule_Adult', 'Purple_Gallinule_Immature', 'Purple_Martin_Adult_male', 'Purple_Martin_Female_OR_juvenile', 'Pygmy_Nuthatch', 'Pyrrhuloxia', 'Reddish_Egret_Dark_morph', 'Reddish_Egret_White_morph', 'Redhead_Breeding_male', 'Redhead_Female_OR_Eclipse_male', 'Red_bellied_Woodpecker', 'Red_breasted_Merganser_Breeding_male', 'Red_breasted_Merganser_Female_OR_immature_male', 'Red_breasted_Nuthatch', 'Red_breasted_Sapsucker', 'Red_Crossbill_Adult_Male', 'Red_Crossbill_Female_OR_juvenile', 'Red_eyed_Vireo', 'Red_headed_Woodpecker_Adult', 'Red_headed_Woodpecker_Immature', 'Red_naped_Sapsucker', 'Red_necked_Grebe_Breeding', 'Red_necked_Grebe_Nonbreeding_OR_juvenile', 'Red_shouldered_Hawk_Adult_', 'Red_shouldered_Hawk_Immature', 'Red_tailed_Hawk_Dark_morph', 'Red_tailed_Hawk_Light_morph_adult', 'Red_tailed_Hawk_Light_morph_immature', 'Red_throated_Loon_Breeding', 'Red_throated_Loon_Nonbreeding_OR_juvenile', 'Red_winged_Blackbird_Female_OR_juvenile', 'Red_winged_Blackbird_Male', 'Ring_billed_Gull_Adult', 'Ring_billed_Gull_Immature', 'Ring_necked_Duck_Breeding_male', 'Ring_necked_Duck_Female_OR_Eclipse_male', 'Ring_necked_Pheasant_Female_OR_juvenile', 'Ring_necked_Pheasant_Male', 'Rock_Pigeon', 'Roseate_Spoonbill', 'Rose_breasted_Grosbeak_Adult_Male', 'Rose_breasted_Grosbeak_Female_OR_immature_male', 'Rosss_Goose', 'Rough_legged_Hawk_Dark_morph', 'Rough_legged_Hawk_Light_morph', 'Royal_Tern', 'Ruby_crowned_Kinglet', 'Ruby_throated_Hummingbird_Adult_Male', 'Ruby_throated_Hummingbird_Female,_immature', 'Ruddy_Duck_Breeding_male', 'Ruddy_Duck_Female_OR_juvenile', 'Ruddy_Duck_Winter_male', 'Ruddy_Turnstone', 'Ruffed_Grouse', 'Rufous_crowned_Sparrow', 'Rufous_Hummingbird_Adult_Male', 'Rufous_Hummingbird_Female,_immature', 'Rusty_Blackbird', 'Sanderling_Breeding', 'Sanderling_Nonbreeding_OR_juvenile', 'Sandhill_Crane', 'Savannah_Sparrow', 'Says_Phoebe', 'Scaled_Quail', 'Scarlet_Tanager_Breeding_Male', 'Scarlet_Tanager_Female_OR_Nonbreeding_Male', 'Scissor_tailed_Flycatcher', 'Semipalmated_Plover', 'Semipalmated_Sandpiper', 'Sharp_shinned_Hawk_Adult_', 'Sharp_shinned_Hawk_Immature', 'Short_billed_Dowitcher', 'Snowy_Egret', 'Snowy_Owl', 'Snow_Bunting_Breeding_adult', 'Snow_Bunting_Nonbreeding', 'Snow_Goose_Blue_morph', 'Snow_Goose_White_morph', 'Solitary_Sandpiper', 'Song_Sparrow', 'Spotted_Sandpiper_Breeding', 'Spotted_Sandpiper_Nonbreeding_OR_juvenile', 'Spotted_Towhee', 'Stellers_Jay', 'Summer_Tanager_Adult_Male', 'Summer_Tanager_Female', 'Summer_Tanager_Immature_Male', 'Surfbird', 'Surf_Scoter_Female_OR_immature', 'Surf_Scoter_Male', 'Swainsons_Hawk_Dark_morph_', 'Swainsons_Hawk_Immature', 'Swainsons_Hawk_Light_morph_', 'Swainsons_Thrush', 'Swallow_tailed_Kite', 'Swamp_Sparrow', 'Tennessee_Warbler', 'Townsends_Solitaire', 'Townsends_Warbler', 'Tree_Swallow', 'Tricolored_Heron', 'Trumpeter_Swan', 'Tufted_Titmouse', 'Tundra_Swan', 'Turkey_Vulture', 'Varied_Thrush', 'Vauxs_Swift', 'Veery', 'Verdin', 'Vermilion_Flycatcher_Adult_male', 'Vermilion_Flycatcher_Female,_immature', 'Vesper_Sparrow', 'Violet_green_Swallow', 'Warbling_Vireo', 'Western_Bluebird', 'Western_Grebe', 'Western_Gull_Adult', 'Western_Gull_Immature', 'Western_Kingbird', 'Western_Meadowlark', 'Western_Sandpiper', 'Western_Screech_Owl', 'Western_Scrub_Jay', 'Western_Tanager_Breeding_Male', 'Western_Tanager_Female_OR_Nonbreeding_Male', 'Western_Wood_Pewee', 'Whimbrel', 'White_breasted_Nuthatch', 'White_crowned_Sparrow_Adult', 'White_crowned_Sparrow_Immature', 'White_eyed_Vireo', 'White_faced_Ibis', 'White_Ibis_Adult', 'White_Ibis_Immature', 'White_tailed_Kite', 'White_throated_Sparrow_Tan_striped_OR_immature', 'White_throated_Sparrow_White_striped', 'White_throated_Swift', 'White_winged_Crossbill_Adult_Male', 'White_winged_Crossbill_Female_OR_juvenile', 'White_winged_Dove', 'White_winged_Scoter_Female_OR_juvenile', 'White_winged_Scoter_Male', 'Wild_Turkey', 'Willet', 'Wilsons_Phalarope_Breeding', 'Wilsons_Phalarope_Nonbreeding,_juvenile', 'Wilsons_Snipe', 'Wilsons_Warbler', 'Winter_Wren', 'Wood_Duck_Breeding_male', 'Wood_Duck_Female_OR_Eclipse_male', 'Wood_Stork', 'Wood_Thrush', 'Wrentit', 'Yellow_bellied_Sapsucker', 'Yellow_billed_Cuckoo', 'Yellow_billed_Magpie', 'Yellow_breasted_Chat', 'Yellow_crowned_Night_Heron_Adult', 'Yellow_crowned_Night_Heron_Immature', 'Yellow_headed_Blackbird_Adult_Male', 'Yellow_headed_Blackbird_Female_OR_Immature_Male', 'Yellow_rumped_Warbler_Breeding_Audubons', 'Yellow_rumped_Warbler_Breeding_Myrtle', 'Yellow_rumped_Warbler_Winter_OR_juvenile_Audubons', 'Yellow_rumped_Warbler_Winter_OR_juvenile_Myrtle', 'Yellow_throated_Vireo', 'Yellow_throated_Warbler', 'Yellow_Warbler']

# Label = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee']


Label = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee', '022.Chuck_will_Widow', '023.Brandt_Cormorant', '024.Red_faced_Cormorant', '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper', '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch', '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '039.Least_Flycatcher', '040.Olive_sided_Flycatcher', '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher', '044.Frigatebird', '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak', '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull', '060.Glaucous_winged_Gull', '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '066.Western_Gull', '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear', '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay', '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher', '080.Green_Kingfisher', '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake', '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole', '096.Hooded_Oriole', '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican', '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit', '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike', '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow', '116.Chipping_Sparrow', '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow', '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow', '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow', '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher', '151.Black_capped_Vireo', '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler', '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler', '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler', '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler', '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren', '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat']


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path =dir_path +"/images/"
test_path = dir_path +"/test/"
data_and_label = []
name_list = []

def Data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range = 0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")
    i = 0
    for name in Label:
        data_path = os.path.join(image_path,name)
        label = Label.index(name)
        for image_name in os.listdir(data_path):
            path = os.path.join(data_path, image_name)
            image = cv2.imread(path)
            image = image.reshape((1,)+image.shape)
            # print(path)
    # image = cv2.imread("D:/FYP_BU/bird/New/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0031_100.jpg")
    # image = image.reshape((1,)+image.shape)

            for batch in datagen.flow(
                image,
                batch_size=16,
                save_to_dir=dir_path+"/test/"+name,
                save_prefix=name,
                save_format="jpg"
            ):
                i +=1
                if i > 20:
                    break


# some script for help me to due with the data
def createDIR():
    for name in Label:
        os.mkdir(test_path+name)

def listname():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for x in os.listdir("images") :
        name_list.append(x)

def move():
    d = dir_path+"/datasets/"
    s = dir_path+"/images/Aberts_Towhee/abc.jpg"
    for x in os.listdir("images") :
        print(x)
        if(os.path.exists(dir_path+"/images/"+x)):
            for y in os.listdir(dir_path+"/images/"+x):
                shutil.copy(dir_path+"/images/"+x+"/"+y, d)
def rename():
    f = open("classes.txt")
    a = f.read().split()
    a = np.array(a).reshape(-1, 2)
    for x in os.listdir("images") :
        if(os.path.exists(dir_path+"/images/"+x)):
            i = 0
            for y in os.listdir(dir_path+"/images/"+x):
                i = i+1
                os.rename((dir_path+"/images/"+x+"/"+y),(dir_path+"/images/"+x+"/"+x+"_"+str(i)+".jpg"))

def pickle_labeName():
    pk = open("name.pickle","wb")
    pickle.dump(Label,pk)
    pk.close()

def load_data_name():
    with open('name.pickle', 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist

# match the data and the label to joblib form
def dataLabel():
    for name in Label:
        data_path = os.path.join(image_path,name)
        label = Label.index(name)

        for image_name in os.listdir(data_path):
            path = os.path.join(data_path, image_name)
            image = cv2.imread(path)
            # iamge = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(224,224))
            # image = np.array(image,dtype=np.float32)
            data_and_label.append([image,label])
            
            # cv2.imshow("image",image)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()
            # print(data_and_label)
    random.shuffle(data_and_label)
    x = []
    y = []
    for image , label in data_and_label:
        x.append(image)
        y.append(label)
    X = np.array(x)
    Y = np.array(y)
    print(X.shape)
    print(type(X))
    print(Y.shape)
    print(type(Y))

    save_x = joblib.dump(X,"x_200_224.pkl")
    save_y = joblib.dump(Y,"y_200_224.pkl")
# dataLabel()









