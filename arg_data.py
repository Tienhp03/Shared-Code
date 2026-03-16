import numpy as np
import pandas as pd

all_time = 150 # frame (time slot)

class CarsPath:
    def __init__(self):

        self.max_time = all_time
        self.csv_path = r"C:\Users\AVSTC\Desktop\2026. Phase Shift_Power Control\env\data\data 01\01_filtered_tracks.csv"

        # Chiều cao theo loại xe
        self.height_map = {
            "Car": 1.5,
            "Truck": 3.5
        }

        self.data = None
        self.unique_ids = None
        self.cars_num = None
        self.max_frame = None # 1 frame = 0.04 second

    def load(self, num_cars: int):

        self.data = pd.read_csv(self.csv_path)

        self.unique_ids = self.data["id"].unique()
        self.unique_ids = self.unique_ids[:num_cars]
        self.cars_num = num_cars

        self.max_frame = self.data["frame"].max()

        print("Selected vehicle IDs:", self.unique_ids)
        print("Max frame:", self.max_frame)

        # Trả về vị trí ban đầu của các xe (frame 0)
        initial_positions = self.get_positions(0)
        return initial_positions

    def get_positions(self, frame: int):

        cars_pos_list = []

        for vid in self.unique_ids:

            car_data = self.data[
                (self.data["id"] == vid) &
                (self.data["frame"] == frame)
            ]

            if not car_data.empty:

                x = car_data["x"].values[0]
                y = car_data["y"].values[0]
                vehicle_type = car_data["vehicle_type"].values[0]

                # Gán chiều cao theo loại xe
                z = self.height_map.get(vehicle_type, 1.5)

                cars_pos_list.append([x, y, z])

        return np.array(cars_pos_list, dtype=np.float32)

    def get_inter_distance(self, frame: int, uav_point: np.ndarray):

        cars_pos = self.get_positions(frame)

        pos_vec = cars_pos - uav_point
        distance = np.linalg.norm(pos_vec, axis=1)

        return cars_pos, distance


# =========================
# TEST
# =========================
if __name__ == "__main__":

    cars = CarsPath()
    cars.load(num_cars=5)

    uav_position = np.array([220, 220, 100], dtype=np.float32)
    frame_test = 150

    positions, distances = cars.get_inter_distance(frame_test, uav_position)

    print("Vehicle positions (x,y,z):\n", positions)
    print("Distances:\n", distances)