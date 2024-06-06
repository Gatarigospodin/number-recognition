from neural_network import load_model
from drawing_app import DrawingApp

# --- Настройки ---
model_filename = "trained_model.pkl"

# --- Загрузка модели ---
nn = load_model(model_filename)
if nn is None:
    print(f"Ошибка: Файл модели не найден: {model_filename}. "
          f"Запустите 'train.py' для обучения модели.")
    exit()

# --- Запуск рисования ---
app = DrawingApp()
app.run(nn)