import os
from PIL import Image
import matplotlib.pyplot as plt

# Путь к основным папкам
root_folder = 'images'
folders = ['test', 'train', 'val']


# Функция для обработки папки и построения графика
def process_folder(folder_path, label):
    sizes = []

    # Считываем все изображения из папки
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = Image.open(os.path.join(folder_path, filename))
                width, height = img.size
                sizes.append((width, height))
            except Exception as e:
                print(f"Ошибка при открытии файла {filename}: {e}")

    # Количество изображений
    print(f"Количество изображений в папке {label}: {len(sizes)}")

    # Строим точечный график
    if sizes:
        widths, heights = zip(*sizes)

        plt.scatter(widths, heights, alpha=0.5, label=label)


# Создаем графики для всех папок
plt.figure(figsize=(10, 6))

for folder in folders:
    folder_path = os.path.join(root_folder, folder)
    process_folder(folder_path, folder)

# Оформление графика
plt.title('Размеры изображений в датасетах')
plt.xlabel('Ширина (px)')
plt.ylabel('Высота (px)')
plt.grid(True)
plt.legend(title='Тип датасета')
plt.show()