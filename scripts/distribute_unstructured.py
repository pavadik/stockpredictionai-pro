import os
import glob
from collections import defaultdict

def main():
    unstructured_dir = r"G:\data2\unstructed"
    target_base = r"G:\data2"
    
    files = glob.glob(os.path.join(unstructured_dir, "SBER*.txt"))
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files to process.")
    
    # We will process file by file to avoid loading everything in memory.
    # But to minimize file open/close, we can group lines by date within each file.
    
    total_lines = 0
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}...")
        lines_by_date = defaultdict(list)
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Example: 02/24/22,070000,186.58,187.54,185.95,185.95,1207620
                parts = line.split(',')
                if len(parts) < 7:
                    continue
                date_str = parts[0]
                lines_by_date[date_str].append(line)
                total_lines += 1
                
        # Write out
        for date_str, lines in lines_by_date.items():
            # Parse MM/DD/YY
            try:
                m_str, d_str, y_str = date_str.split('/')
                m = int(m_str)
                d = int(d_str)
                y = 2000 + int(y_str)
                
                # Target path: G:\data2\YYYY\M\D\SBER\M1\data.txt
                target_dir = os.path.join(target_base, str(y), str(m), str(d), "SBER", "M1")
                os.makedirs(target_dir, exist_ok=True)
                
                target_file = os.path.join(target_dir, "data.txt")
                
                # Append or write. Let's append to be safe.
                # Usually we want to write cleanly, but there might be overlaps? 
                # Let's read existing lines if it exists to avoid duplicates.
                existing_lines = set()
                if os.path.exists(target_file):
                    with open(target_file, "r", encoding="utf-8") as tf:
                        existing_lines = set(l.strip() for l in tf if l.strip())
                
                new_lines = []
                for l in lines:
                    if l not in existing_lines:
                        new_lines.append(l)
                        existing_lines.add(l)
                
                if new_lines:
                    with open(target_file, "a", encoding="utf-8") as tf:
                        for l in new_lines:
                            tf.write(l + "\n")
            except Exception as e:
                print(f"Error parsing date {date_str}: {e}")

    print(f"Done. Processed {total_lines} lines.")

if __name__ == "__main__":
    main()
