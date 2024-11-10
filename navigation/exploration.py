import numpy as np
from queue import Queue

class FrontierExplorer:
    def __init__(self, mapper, start_position):
        """
        mapper: The Mapper instance.
        start_position: Starting position of the robot in the form (x, y).
        """
        self.mapper = mapper
        self.current_position = start_position

    def detect_frontiers(self):
        """
        Use the Mapper to detect frontier cells.
        """
        return self.mapper.get_frontiers()

    def select_next_frontier(self, frontiers):
        """
        Select the closest frontier from the current position.
        """
        if not frontiers:
            return None
        return min(frontiers, key=lambda p: np.linalg.norm(np.array(p) - np.array(self.current_position)))

    def navigate_to(self, goal):
        """
        Move towards the goal (frontier).
        """
        print(f"Navigating to frontier: {goal}")
        # Placeholder navigation logic. Replace with real navigation planning.
        self.current_position = goal

    def explore(self):
        """
        Main exploration loop.
        """
        while True:
            frontiers = self.detect_frontiers()
            if not frontiers:
                print("No more frontiers found. Exploration complete.")
                break

            next_frontier = self.select_next_frontier(frontiers)
            if next_frontier:
                self.navigate_to(next_frontier)

                # Simulate mapping new areas once the robot reaches the frontier.
                new_points = self.simulate_scan_at_frontier(next_frontier)
                self.mapper.update_grid(new_points, state=0)  # Mark as free

            # Print grid for debugging
            self.mapper.print_grid()

    def simulate_scan_at_frontier(self, position):
        """
        Simulate scanning new points at the given frontier position.
        This simulates the scanning sensor (e.g., LIDAR).
        """
        x, y = position
        new_points = [
            (x + dx * self.mapper.resolution, y + dy * self.mapper.resolution)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        return new_points

# Example Usage
if __name__ == "__main__":
    # Define map size and resolution
    grid_size = (20, 20)  # 20x20 grid
    resolution = 0.5      # Each cell represents 0.5m

    # Initialize the Mapper and the Frontier Explorer
    mapper = Mapper(grid_size, resolution)
    explorer = FrontierExplorer(mapper, start_position=(5, 5))

    # Update the map with some obstacles (for example purposes)
    obstacles = [(4, 4), (5, 4), (6, 4), (4, 5)]
    mapper.update_grid(obstacles, state=1)  # Mark these points as occupied

    # Start the exploration
    explorer.explore()
