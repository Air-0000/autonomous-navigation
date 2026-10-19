"""Comprehensive Unit and Integration Tests for Biologically-Grounded Cognitive Navigation."""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cognition.cognitive_map import (
    HeadDirectionNetwork,
    GridCellModule,
    PlaceCellNetwork,
    CognitiveMap,
)


def test_head_direction_ring_attractor():
    """Test HD ring attractor integrates angular velocity accurately."""
    hd_net = HeadDirectionNetwork(n_neurons=60)
    hd_net.reset_heading(0.0)
    assert abs(hd_net.get_heading()) < 0.1

    # Rotate 90 degrees (pi/2 rad) at 1.0 rad/s over pi/2 seconds
    dt = 0.02
    steps = int((np.pi / 2) / dt)
    for _ in range(steps):
        hd_net.update(angular_vel=1.0, dt=dt)

    decoded_heading = hd_net.get_heading()
    expected_heading = np.pi / 2
    assert abs(decoded_heading - expected_heading) < 0.15, f"Expected ~{expected_heading:.2f}, got {decoded_heading:.2f}"


def test_grid_cell_cann_path_integration():
    """Test 2D CANN grid cell module shifts activity bump proportionally to velocity."""
    grid = GridCellModule(grid_size=24, spatial_wavelength=1.0)
    init_phase = grid.get_phase()
    
    # Move forward along X axis at 0.5 m/s for 1.0 s -> 0.5 m displacement
    dt = 0.05
    for _ in range(20):
        grid.update(velocity=(0.5, 0.0), heading=0.0, dt=dt)
        
    final_phase = grid.get_phase()
    phase_shift_x = (final_phase[0] - init_phase[0]) % 1.0
    assert abs(phase_shift_x - 0.5) < 0.12, f"Expected phase shift ~0.5m, got {phase_shift_x:.2f}"


def test_place_cell_emergence_and_landmark_association():
    """Test place cell activation and Hebbian visual landmark drift correction."""
    cog_map = CognitiveMap()
    
    # Move in square: (0,0) -> (2,0) -> (2,2) -> (0,2) -> (0,0)
    dt = 0.1
    for _ in range(20):  # Forward 2m
        cog_map.update_motion(linear_vel=(1.0, 0.0), angular_vel=0.0, dt=dt)
        
    # Associate a landmark at (2.0, 0.0)
    cog_map.correct_with_visual_landmark(landmark_name="red_pillar", relative_dist=0.5, relative_bearing=0.0)
    assert "red_pillar" in cog_map.place_net.landmark_memory
    
    # Introduce artificial cumulative drift
    cog_map.current_x += 0.8  # 0.8m drift error
    drift_before = cog_map.current_x
    
    # Re-observe landmark at (2.0, 0.0)
    cog_map.correct_with_visual_landmark(landmark_name="red_pillar", relative_dist=0.5, relative_bearing=0.0)
    drift_after = cog_map.current_x
    
    # Drift must be significantly reduced towards true location
    assert abs(drift_after - 2.0) < abs(drift_before - 2.0), "Landmark re-observation failed to correct position drift!"


def test_vector_navigation_command():
    """Test cognitive goal-directed vector planning outputs correct steering."""
    cog_map = CognitiveMap()
    cog_map.current_x = 0.0
    cog_map.current_y = 0.0
    cog_map.hd_net.reset_heading(0.0)
    
    # Goal at (5.0, 0.0) right ahead: should command straight forward velocity
    v, w = cog_map.compute_navigation_command(goal_pos=(5.0, 0.0))
    assert v > 0.3, "Linear velocity should be positive towards ahead goal"
    assert abs(w) < 0.1, "Angular velocity should be near zero for ahead goal"
    
    # Goal at (0.0, 5.0) to the left (pi/2 rad): should command counter-clockwise turn
    v, w = cog_map.compute_navigation_command(goal_pos=(0.0, 5.0))
    assert w > 0.5, "Angular velocity should steer left towards goal"


if __name__ == "__main__":
    test_head_direction_ring_attractor()
    print("✅ Head Direction Attractor Test Passed")
    test_grid_cell_cann_path_integration()
    print("✅ Grid Cell CANN Path Integration Test Passed")
    test_place_cell_emergence_and_landmark_association()
    print("✅ Place Cell Landmark Drift Reset Test Passed")
    test_vector_navigation_command()
    print("✅ Vector Navigation Command Test Passed")
    print("\n🎉 ALL BRAIN-INSPIRED COGNITIVE NAVIGATION TESTS PASSED!")
