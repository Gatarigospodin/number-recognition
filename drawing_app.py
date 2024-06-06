import pygame
import numpy as np


class DrawingApp:
    def __init__(self, width=560, height=280):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Рисовалка чисел")

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.button_color = (200, 200, 200)

        self.brush_size = 10
        self.font = pygame.font.Font(None, 24)

        self.canvas = [[0 for _ in range(28)] for _ in range(28)]

        self.button_width = 100
        self.button_height = 30
        self.button_x = self.width - self.button_width - 20
        self.button_y = self.height - self.button_height - 20

        self.drawing = False
        self.predictions = None  # Изначально нет предсказаний

    def run(self, nn=None):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Левая кнопка - рисование
                        self.drawing = True
                        if self.button_x <= event.pos[0] <= self.button_x + self.button_width and \
                                self.button_y <= event.pos[1] <= self.button_y + self.button_height:
                            self.clear_canvas()
                    elif event.button == 3:  # Правая кнопка - стирание
                        self.handle_mouse_motion(event.pos, erase=True)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Левая кнопка
                        self.drawing = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        self.handle_mouse_motion(event.pos)
                    elif event.buttons[2]:  # Правая кнопка зажата
                        self.handle_mouse_motion(event.pos, erase=True)

            self.draw(nn)
            pygame.display.flip()

        pygame.quit()

    def handle_mouse_motion(self, pos, erase=False):
        x, y = pos
        row = y // self.brush_size
        col = x // self.brush_size

        if 0 <= row < 28 and 0 <= col < 28:
            if erase:
                self.canvas[row][col] = 0  # Стирание
            else:
                for i in range(2):
                    for j in range(2):
                        if 0 <= row + i < 28 and 0 <= col + j < 28:
                            self.canvas[row + i][col + j] = 1

    def clear_canvas(self):
        with (open("data.txt", "w")) as f:
            f.write(str(np.array(self.canvas).reshape(1, 784)))
        self.canvas = [[0 for _ in range(28)] for _ in range(28)]

    def draw(self, nn):
        self.screen.fill(self.white)

        for row in range(28):
            for col in range(28):
                if self.canvas[row][col] == 1:
                    pygame.draw.rect(self.screen, self.black,
                                     (col * self.brush_size, row * self.brush_size,
                                      self.brush_size, self.brush_size))

        # Предсказания только если есть рисование
        if any(1 in row for row in self.canvas) and nn:
            input_image = np.array(self.canvas).reshape(1, 784)
            self.predictions = nn.forward(input_image)[0]

        # Отображение предсказаний
        if self.predictions is not None:
            for i in range(10):
                text = self.font.render(f"{i}: {self.predictions[i] * 100:.1f}%", True, self.black)
                self.screen.blit(text, (self.width - self.button_width - 180, i * 30))

        pygame.draw.rect(self.screen, self.button_color,
                         (self.button_x, self.button_y, self.button_width, self.button_height))
        button_text = self.font.render("Очистить", True, self.black)
        self.screen.blit(button_text, (self.button_x + 15, self.button_y + 7))


# Запуск рисования, если файл запущен напрямую
if __name__ == "__main__":
    app = DrawingApp()
    app.run()
