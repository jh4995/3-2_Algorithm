import os

DATA_PATH = r""

if __name__ == "__main__":
    
    folder_count = 0
    json_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(DATA_PATH):
        folder_count += 1  # 현재 폴더 포함
        for f in files:
            file_path = os.path.join(root, f)
            total_size += os.path.getsize(file_path)
            if f.endswith('.json'):
                json_count += 1
    print(f"전체 폴더 개수: {folder_count}")
    print(f"JSON 파일 개수: {json_count}")

    print(f"전체 파일 크기: {total_size} 바이트")
