#!/bin/bash

# Navigate to bin, exit the script if the directory doesn't exist to prevent errors
cd bin || exit

# Define the list of files as a bash array
X=("ex3_random.py" "ex3-v1.py" "ex3-v2.py" "ex3-v3.py" "ex3-v4.py")
# X=("ex3_random.py")
# Loop through each item in the array
for version in "${X[@]}"; do
    echo "Playing version: $version"
    # Copy the current version to ex3.py
    cp "../my/versions/$version" ex3.py
    
    # Run the check script and redirect stdout to the output file
    python3 ex3_check.py > "../tests/liel/output/${version}.txt"
    
    # Move the Solution.txt to the results folder
    mv Solution.txt "../tests/liel/results/${version}.txt"
done

# Return to the parent directory
cd ..