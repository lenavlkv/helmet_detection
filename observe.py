import os
import numpy as np
from collections import Counter

def parse_annotations(folder_path):
    counts_1 = 0  # Количество 1 (каски)
    counts_0 = 0  # Количество 0 (без каски)
    areas = []  # Список всех площадей bounding box'ов (головы или каски)
    helmet_areas = []  # Площади bounding box'ов с касками (где метка 1)


    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):

            with open(os.path.join(folder_path, file_name), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.strip().split()
                    label = int(data[0])  # 1 или 0 (наличие каски)
                    x_center, y_center, width, height = map(float, data[1:])

                    # Вычисляем площадь для bounding box (головы или каски)
                    area = width * height
                    areas.append(area)

                    # Если каска есть, добавляем площадь в helmet_areas
                    if label == 1:
                        counts_1 += 1
                        helmet_areas.append(area)
                    if label == 0:
                        counts_0 += 1

    # Вычисляем среднюю площадь касок, если таковые есть
    if counts_1 > 0:
        avg_helmet_area = np.mean(helmet_areas)
    else:
        avg_helmet_area = 0  # Если касок нет, ставим 0

    # Вычисляем моду площади (наиболее часто встречающееся значение)
    area_counts = Counter(areas)
    mode_area, _ = area_counts.most_common(1)[0]

    return counts_1, counts_0, areas, avg_helmet_area, mode_area


# Функция для обработки всех папок
def process_dataset():
    # Папки с аннотациями
    folders = {
        'train': 'labels/train',
        'test': 'labels/test',
        'valid': 'labels/val'
    }

    # Суммарные счетчики для всех папок
    total_counts_1 = 0
    total_counts_0 = 0
    total_areas = []  # Все площади по всем папкам
    total_helmet_areas = []  # Все площади касок по всем папкам

    # Проходим по всем папкам
    for folder_name, folder_path in folders.items():
        # Получаем статистику по каждой папке
        counts_1, counts_0, areas, avg_helmet_area, mode_area = parse_annotations(folder_path)

        # Добавляем в суммарные счетчики
        total_counts_1 += counts_1
        total_counts_0 += counts_0

        # Добавляем площади в общий список
        total_areas.extend(areas)
        total_helmet_areas.extend(total_helmet_areas)

        # Выводим результаты по каждой папке
        print(f"Folder: {folder_name}")
        print(f"  Counts of 1 (helmets): {counts_1}")
        print(f"  Counts of 0 (no helmet): {counts_0}")
        print(f"  Percentage of 1: {(counts_1 / (counts_1 + counts_0)) * 100:.2f}%")
        print(f"  Percentage of 0: {(counts_0 / (counts_1 + counts_0)) * 100:.2f}%")
 #       print(f"  Average helmet area: {avg_helmet_area:.4f} square units")
        print(f"  Mode of areas: {mode_area:.4f} square units")
        print()


    # Рассчитываем моду для всех данных
    area_counts = Counter(total_areas)
    global_mode_area, _ = area_counts.most_common(1)[0]

    # Вычисляем процентное соотношение для суммарных данных
    total_percentage_1 = (total_counts_1 / (total_counts_1 + total_counts_0)) * 100
    total_percentage_0 = (total_counts_0 / (total_counts_1 + total_counts_0)) * 100

    # Выводим суммарные результаты
    print(f"Total counts of 1 (helmets): {total_counts_1}")
    print(f"Total counts of 0 (no helmet): {total_counts_0}")
    print(f"Total percentage of 1: {total_percentage_1:.2f}%")
    print(f"Total percentage of 0: {total_percentage_0:.2f}%")
 #   print(f"Global average helmet area: {avg_helmet_area_all:.4f} square units")
    print(f"Global mode of areas: {global_mode_area:.4f} square units")

# Основная функция
process_dataset()