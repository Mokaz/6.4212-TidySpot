#!/bin/bash

# Array of scenario files
scenario_files=("objects/cracker.yaml" "objects/meat.yaml" "objects/soup.yaml" "objects/mustard.yaml")
# Loop through each file in the array
for scenario_file in "${scenario_files[@]}"; do
  echo "Processing $scenario_file..."
  python main.py --grasp_type anygrasp --perception_type ground_truth --scenario "$scenario_file"

  # Check the exit status of the command
  if [ $? -ne 0 ]; then
    echo "Error: Command failed for $scenario_file. Exiting."
    exit 1
  fi

  echo "$scenario_file processed successfully."
done

echo "All scenario files processed."