import os
import json
import sys

def calculate_bounding_boxes(pcd_filepath):
    """
    Parses a single ASCII PCD file and calculates the bounding box for each labeled object.

    Args:
        pcd_filepath (str): The path to the input .pcd file.

    Returns:
        dict: A dictionary containing bounding box data, or None if the file cannot be processed.
    """
    points_by_label = {}

    try:
        with open(pcd_filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{pcd_filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{pcd_filepath}': {e}")
        return None


    data_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if data_started:
            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                label = parts[-1]

                if label not in points_by_label:
                    points_by_label[label] = []
                points_by_label[label].append((x, y, z))
            except (ValueError, IndexError):
                # This line is likely not a valid point, skip it
                continue

        if line.startswith('DATA ascii'):
            data_started = True

    if not points_by_label:
        return None # Return None if no valid labeled points were found

    bounding_boxes = {}
    for label, points in points_by_label.items():
        min_x = min_y = min_z = sys.float_info.max
        max_x = max_y = max_z = -sys.float_info.max

        for x, y, z in points:
            min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
            max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
        
        bounding_boxes[label] = {
            'min_corner': [min_x, min_y, min_z],
            'max_corner': [max_x, max_y, max_z]
        }
    
    return bounding_boxes

def process_directory(input_root, output_root):
    """
    Walks through the input directory, processes all .pcd files, and saves
    bounding box .json files to a mirrored directory structure in the output_root.
    """
    print(f"Starting Scan...\nInput Dir:  '{input_root}'\nOutput Dir: '{output_root}'\n")
    
    file_count = 0
    # os.walk is a generator that explores a directory tree top-down
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            # Check if the file is a PCD file
            if filename.lower().endswith('.pcd'):
                input_pcd_path = os.path.join(dirpath, filename)
                
                # --- Create the mirrored output path ---
                # 1. Get the relative path from the input root
                relative_path = os.path.relpath(dirpath, input_root)
                # 2. Create the corresponding output directory
                output_dir = os.path.join(output_root, relative_path)
                # 3. Ensure this output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # 4. Create the final JSON output filename
                base_filename, _ = os.path.splitext(filename)
                output_json_path = os.path.join(output_dir, base_filename + '.json')
                
                print(f"Processing: {input_pcd_path}")
                
                # Calculate the bounding boxes for the current file
                bounding_boxes = calculate_bounding_boxes(input_pcd_path)
                
                if bounding_boxes:
                    # Save the result to the new JSON file
                    with open(output_json_path, 'w') as f:
                        json.dump(bounding_boxes, f, indent=4)
                    print(f" -> Saved to: {output_json_path}\n")
                    file_count += 1
                else:
                    print(f" -> Skipped (no valid data or labels found).\n")

    print(f"Finished. Processed and created {file_count} JSON files.")


# --- HOW TO USE ---
if __name__ == "__main__":
    # 1. Specify the root folder containing your PCD files
    INPUT_FOLDER = "All-Datasets/v3-infingen-scene-mdm-human"

    # 2. Specify the root folder where you want the JSON files to be saved
    OUTPUT_FOLDER = "BB/v3-infingen-scene-mdm-human"

    # # --- Create a dummy file structure for demonstration ---
    # print("Setting up a dummy directory structure for demonstration...")
    # # Room 1 data
    # os.makedirs(os.path.join(INPUT_FOLDER, "scene_01", "room_101"), exist_ok=True)
    # with open(os.path.join(INPUT_FOLDER, "scene_01", "room_101", "scan1.pcd"), "w") as f:
    #     f.write("""DATA ascii
    #     1.0 1.0 1.0 Table_A
    #     1.5 1.0 1.0 Table_A
    #     2.0 2.0 2.0 Chair_B
    #     2.5 2.5 2.5 Chair_B
    #     """)
    # # Room 2 data
    # os.makedirs(os.path.join(INPUT_FOLDER, "scene_01", "room_102"), exist_ok=True)
    # with open(os.path.join(INPUT_FOLDER, "scene_01", "room_102", "scan2.pcd"), "w") as f:
    #     f.write("""DATA ascii
    #     -5.0 -5.0 3.0 Window_C
    #     -5.5 -5.5 3.5 Window_C
    #     """)
    # # Top-level file
    # with open(os.path.join(INPUT_FOLDER, "overview.pcd"), "w") as f:
    #     f.write("""DATA ascii
    #     100 100 100 Building_D
    #     """)
    # print("Dummy structure created.\n" + "-"*20)
    # # --- End of dummy structure creation ---


    # Run the main processing function
    process_directory(INPUT_FOLDER, OUTPUT_FOLDER)

    # 2nd folder 
    INPUT_FOLDER = "All-Datasets/v5-infingen-scene-humanML3d-human"
    OUTPUT_FOLDER = "BB/v5-infingen-scene-humanML3d-human"
    process_directory(INPUT_FOLDER, OUTPUT_FOLDER)