from final.display_stolen_goods import Stolen_Goods
from final.gen_dataset import DataSet
from final.map_class import Map_Location

bad_MMSI = [338356427.0, 338226496.0, 367459470, 368104650, 367096960, 368128140, 338168958,
            367393710, 367638310, 338203701, 338229321, 368124390, 338229941, 8704092, 368128140, 367526270, 367072720,
            368006440, 367105250, 338356627, 367101090, 367518040, 338412448, 338459234, 338312689, 368178890,
            367767060, 338334752, 338359496, 338336234, 338926089, 338413783, 338399049, 338386000, 338402566.0,
            368037170.0, 367767060.0, 367722440.0, 367613470.0,
            338399049.0, 338386000.0, 368091370.0, 367092940.0, 91800312.0,
            91800534.0, 91800557.0, 91800425.0, 367177060.0, 366866750.0,
            338144528.0, 338362611.0, 338438928.0, 368325670.0, 338486893.0,
            338489737.0, 368321520.0, 338485667.0, 338448489.0, 367111210.0,
            338489747.0, 338066383.0, 368128790.0, 338169001.0, 338482296.0,
            338485786.0, 338350046.0, 338419562.0, 338475816.0, 368318450.0,
            338371638.0, 338480232.0, 368280420.0, 368265580.0, 368114380.0,
            367075540.0, 316003440.0, 338095648.0, 338312689.0, 338459234.0,
            338412448.0, 368178890.0, 338334752.0, 338926089.0, 338413783.0,
            338191082.0
            ]

montuk = Map_Location(72.283333, 41.256567, 0.016667, 550, 0.016667, 730,
                      "/Users/olivereielson/Desktop/13209-11-1995.jpg", "Montuk", square_tiles=True)

montuk.add_exclusion([41.080349, 71.941680, 41.075006, 71.931822])
montuk.add_exclusion([41.074323, 71.942715, 41.050378, 71.895648])
montuk.add_exclusion([41.034126, 72.190836, 41.020868, 72.163012])
montuk.add_exclusion([41.15585, 72.24802, 41.15441, 72.23412])

block_island = Map_Location(71.86222222, 41.38333333, 0.016667, 275, 0.016667, 365,
                            "/Users/olivereielson/Desktop/13215-7-2023.jpg", "Block Island", square_tiles=True)
block_island.add_exclusion([41.199359, 71.598857, 41.181609, 71.560790])
block_island.add_exclusion([41.37512, 71.52909, 41.43013, 71.45829])
block_island.add_exclusion([41.04555556, 71.65, 40.5, 71])
block_island.add_exclusion([40.985, 71.75, 40.5, 71])

buzzards_bay = Map_Location(70.98888889, 41.77222222, 0.016667, 545, 0.016667, 725,
                            "/Users/olivereielson/Desktop/13230-01-2017.jpg", "Buzzards Bay", square_tiles=True)

long_island = Map_Location(73.13333333, 41.36666667, 0.016667, 275, 0.016667, 365,
                           "/Users/olivereielson/Desktop/12354-12-2018.jpg", "Long Island", square_tiles=True)

long_island.add_exclusion([40.95183, 72.21422, 41.14714, 72.56579])
long_island.add_exclusion([41.04429, 72.17926, 41.00292, 72.21494])

test_all = Map_Location(74.15277778, 45.00000000, 0.16667, 275, 0.16667, 390,
                        "/Users/olivereielson/Desktop/13006-04-2009.jpg",
                        "all", square_tiles=False)

lat_width = abs(73.6248779296875 - 73.619384765625)
lon_width = abs(41.00477542222947 - 41.000629848685385)

data = Stolen_Goods(73.10319, 41.15209, 999, 685, lat_width, lon_width, 180, 180, image_file_paths="grids/",
                    unwanted_mmsi=bad_MMSI)
data.add_exclusion([41.080349, 71.941680, 41.075006, 71.931822])
data.add_exclusion([41.074323, 71.942715, 41.050378, 71.895648])
data.add_exclusion([41.034126, 72.190836, 41.020868, 72.163012])
data.add_exclusion([41.15585, 72.24802, 41.15441, 72.23412])
data.add_exclusion([40.95183, 72.21422, 41.14714, 72.56579])
data.add_exclusion([41.04429, 72.17926, 41.00292, 72.21494])
data.add_exclusion([41.199359, 71.598857, 41.181609, 71.560790])
data.add_exclusion([41.37512, 71.52909, 41.43013, 71.45829])
data.add_exclusion([41.04555556, 71.65, 40.5, 71])
data.add_exclusion([40.985, 71.75, 40.5, 71])

data.generate()
data.show_images()

# data = DataSet(1000, "all.csv", "grids/", output_path="data_labels.csv",
#                locations=[montuk],
#                unwanted_mmsi=bad_MMSI, regenerate=False, )
# #
# data.generate()
# data.show_images()
# # data.draw_exclusion()

# data.test_model(threshold=0.5)

# data.draw_predictions()


# ON EDGE
# GUNSMOKE
# PROVIDER
# PRIME RATE
# OCEAN VUE
# SHEILA
# TUNAS DREAM II
# HOOKED UP
# HIT & RUN
# REALM
# SIMON SEZ
# 'ELLEN JEAN
# OCEAN ROSE
# JOY SEA
##NEVER ENOUGH
# MACKEREL SKY
# BOTTOM LINE
# TUNA MILL 3
# JAKAMO 2
# HUZZAH
# REEL TRAFFIC
# ALTHEA
# BLACK ROCK
# THE EXPERIENCE


# ERICA KNIGHT
# STAR
# STELLA MARIS
# PROVIDER
# PRIME RATE
# HARVEST MOON
# PROVIDER II
# DOUBLE HEADER
# RELENTLESS
# ONWA33NT
# F/V BLUE SEA
# Prince Rupert, BC
# PACER
# ATLANTIC PEARL
# DREAM WARRIOR
# MISTE ROSE
# SCOOTER
# MARY ANNE
# GREAT ESCAPE
# OLD COOT
# GARLIC KNOT
# LOOSE CANNON
# ANNA MARY
# SWEET MISERY
# NOAA GLORIA MICHELLE
# GULF STREAM
# PREY DRIVE
# DISRUPTOR
# STILL FISHING
# ARGO
# KARENB
# ODONTOBLAST
# FORTITUDE
# GRATTITUDE
# STORSKIP
# EDWARD&JOSEPH
# MARTHA ELIZABETH
# nan
# PATRIARCH
# MISS SHAUNA
# AMANDA J I
# INGRID ANN
# FV SEA SPRITE
# SALTY DOG
# ALLISON AN LISA
# JEANETTE T
# HANA-LI
# 29RDC
# BLACK EARL
# TERNTO
# HILLARY ANN
# PASQUE
#
#
#
#
#
