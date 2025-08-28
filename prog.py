import pygame
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PIL import Image, ImageOps
# Структура модели
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# Основное приложение
class DigitRecognizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 500))  # Размер рамки программы
        pygame.display.set_caption("MNIST Digit Recognizer")
        self.clock = pygame.time.Clock()
        
        # Загрузка модели
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MyCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load("my_model_CNN.pth", map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            print("Ошибка: Файл my_model_CNN.pth не найден!")
            return
        
        # Поверхность для рисования
        self.canvas = pygame.Surface((280, 280))
        self.canvas.fill((255, 255, 255))  # Белый фон
        
        # Параметры
        self.drawing = False
        self.last_pos = None
        self.probs = [0] * 10
        self.predicted = "None"
        self.font = pygame.font.SysFont("arial", 20)

    def draw_line(self, pos):
        if self.last_pos is not None:
            pygame.draw.line(self.canvas, (0, 0, 0), self.last_pos, pos, 10)
        self.last_pos = pos

    def clear_canvas(self):
        self.canvas.fill((255, 255, 255))
        self.probs = [0] * 10
        self.predicted = "None"

    def predict(self):
        # Получаем numpy-массив из canvas
        canvas_array = pygame.surfarray.array3d(self.canvas)
        canvas_array = canvas_array[:, :, 0].T  # Берём только один канал

        # Конвертируем в Pillow Image и инвертируем (MNIST: чёрное = 1)
        img = Image.fromarray(canvas_array, mode="L")
        img = Image.eval(img, lambda x: 255 - x)  # Инверсия

        # Обрезаем белые края (bounding box)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Увеличиваем до 20x20 внутри белого фона 28x28
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        new_img = Image.new("L", (28, 28), 0)  # Чёрный фон (как инвертированное MNIST)
        new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

        # Преобразуем в тензор
        img_array = np.array(new_img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted = np.argmax(probs)

        self.probs = probs * 100
        self.predicted = str(predicted)


        
        # Предсказание
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted = np.argmax(probs)
        
        self.probs = probs * 100  # Масштабируем для отображения
        self.predicted = str(predicted)

    def draw_bars(self):
        bar_width = 40
        spacing = 15    
        max_height = 70
        for i, prob in enumerate(self.probs):
            x = i * (bar_width + spacing) + 40
            y = 400
            height = min(prob, 100) * max_height / 100
            pygame.draw.rect(self.screen, (0, 255, 0), (x, y - height, bar_width, height))
            label = self.font.render(str(i), True, (0, 0, 0))
            self.screen.blit(label, (x + bar_width / 2 - 5, y + 5))
            prob_text = self.font.render(f"{prob:.1f}%", True, (0, 0, 0))
            self.screen.blit(prob_text, (x, y - height - 20))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        x, y = event.pos
                        canvas_x, canvas_y = x - 160, y - 20
                        if 0 <= canvas_x <= 280 and 0 <= canvas_y <= 280:
                            self.drawing = True
                            self.draw_line((canvas_x, canvas_y))
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                        self.last_pos = None
                        self.predict()
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    x, y = event.pos
                    canvas_x, canvas_y = x - 160, y - 20
                    if 0 <= canvas_x <= 280 and 0 <= canvas_y <= 280:
                        self.draw_line((canvas_x, canvas_y))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.clear_canvas()
            
            # Отрисовка
            self.screen.fill((200, 200, 200))  # Серый фон
            self.screen.blit(self.canvas, (160, 20))  # Центрировано
            pygame.draw.rect(self.screen, (0, 0, 0), (160, 20, 280, 280), 2)  # Рамка холста
            self.draw_bars()
            pred_text = self.font.render(f"Predicted: {self.predicted}", True, (0, 0, 0))
            self.screen.blit(pred_text, (220, 450))
            
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    app = DigitRecognizer()
    app.run()