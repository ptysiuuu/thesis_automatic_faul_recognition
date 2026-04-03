import os

def check_integrity(base_path):
    print(f"Analizowanie ścieżki: {base_path}")
    
    if not os.path.exists(base_path):
        print("BŁĄD: Ścieżka nie istnieje!")
        return

    all_actions = [d for d in os.listdir(base_path) if d.startswith('action_')]
    all_actions.sort(key=lambda x: int(x.split('_')[1]))
    
    total_folders = len(all_actions)
    valid_actions = 0
    corrupted_actions = 0
    missing_videos = 0
    
    corrupted_list = []

    for action in all_actions:
        action_path = os.path.join(base_path, action)
        # Pobierz pliki .mp4 wewnątrz akcji
        videos = [f for f in os.listdir(action_path) if f.endswith('.mp4')]
        
        if not videos:
            missing_videos += 1
            corrupted_list.append(f"{action}: Brak plików .mp4")
            continue
            
        # Sprawdź, czy którykolwiek plik ma więcej niż 2KB
        is_valid = False
        for v in videos:
            fsize = os.path.getsize(os.path.join(action_path, v))
            if fsize > 2048: # Więcej niż 2KB
                is_valid = True
                break
        
        if is_valid:
            valid_actions += 1
        else:
            corrupted_actions += 1
            corrupted_list.append(f"{action}: Wszystkie klipy uszkodzone (1KB)")

    print("\n--- RAPORT INTEGRALNOŚCI ---")
    print(f"Znalezionych folderów action_*: {total_folders}")
    print(f"Akcje z poprawnymi wideo (>2KB): {valid_actions}")
    print(f"Akcje całkowicie uszkodzone:     {corrupted_actions}")
    print(f"Akcje bez żadnych plików .mp4:   {missing_videos}")
    print("----------------------------")
    
    if corrupted_list:
        print("\nPierwszych 10 problematycznych akcji:")
        for item in corrupted_list[:10]:
            print(f" - {item}")

if __name__ == "__main__":
    # Ścieżka na Athenie
    path = "/net/tscratch/people/plgaszos/SoccerNet_Data/train_720p"
    check_integrity(path)
