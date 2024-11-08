# SUPER_CLASS_MAP = {
#     "vehicle": ["Car", "PoliceCar", "LongVehicle"],
#     "pedestrian": ["ScooterRider", "Pedestrian",
#                        "BicycleRider", "MotorcyleRider",
#                        "Scooter", "Motorcycle", "Child",
#                        "RoadWorker", "Bicycle"],
#     "truck": ["TrashCan", "Van", "Bus",
#               "ConcreteTruck", "Truck",
#               "FireHydrant", "ConstructionCart"],
# }
# [smallest object, ..., largest object in the super class category]
SUPER_CLASS_MAP = {
    "vehicle": ["LongVehicle", "Car", "PoliceCar"],
    "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter",
                   "ScooterRider", "Motorcycle", "MotorcyleRider",
                   "BicycleRider"],
    "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"],
}
# Car, [Pedestrian, Scooter, Motorcycle, Bicycle], [Truck, Van, TrashCan, ConcreteTruck, Bus]