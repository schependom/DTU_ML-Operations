# After running
#
# *      uvr python -m cProfile -o analysis/profiling/profile.prof src/ml_ops/train.py
#
# this script can be used to analyze the profiling results.
#
# Run with:
#
#       uvr python analysis/profiling/analyze.py

import os
import pstats
import subprocess
from pathlib import Path

base = "analysis/profiling"

# Get this folder and append to path
profile_file = f"{base}/profile.prof"
p = pstats.Stats(profile_file)
p.sort_stats("cumulative").print_stats(10)

# Export visualization with snakeviz
viz_dir = Path(f"{base}/viz")
viz_dir.mkdir(exist_ok=True)

print("\n" + "=" * 60)
print("Generating snakeviz visualization...")
print("=" * 60 + "\n")

# Run snakeviz to generate HTML output
output_file = viz_dir / "profile_visualization.html"
try:
    # Use snakeviz to open and save the visualization
    # snakeviz will create an HTML file that can be opened in a browser
    subprocess.run(["snakeviz", "-s", profile_file], cwd=os.getcwd(), check=True)
    print(f"✓ Visualization server started!")
    print(f"  The profile data is located at: {profile_file}")
    print(f"  Open the browser to view the interactive visualization")
except FileNotFoundError:
    print("ERROR: snakeviz is not installed!")
    print("Install it with: uv sync --dev")
except Exception as e:
    print(f"ERROR: {e}")

# tottime
#   is the total time spent in the function excluding time spent in subfunctions.
#
# cumtime
#   is the total time spent in the function including time spent in subfunctions.
#
# Therefore, cumtime is always greater than tottime.
