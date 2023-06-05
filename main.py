import pygame
import numpy as np
from tensorflow.keras.models import load_model

# Pygame initialization
pygame.init()
WIDTH, HEIGHT = 280, 280
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwritten Digit Recognition")

# Load the trained model
model = load_model('mnist.h5')

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Variables
drawing = False
last_pos = (0, 0)
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)

# Function to preprocess and predict the drawn digit
def predict_digit(image):
    # Preprocess the image
    temp_surface = pygame.Surface((WIDTH, HEIGHT))
    temp_surface.blit(image, (0, 0))
    temp_surface = pygame.transform.scale(temp_surface, (28, 28))
    image = pygame.surfarray.array2d(temp_surface)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Perform inference
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return digit

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            pygame.draw.circle(canvas, WHITE, event.pos, 10)
            last_pos = event.pos
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.line(canvas, WHITE, last_pos, event.pos, 20)
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            digit = predict_digit(canvas)

            # Display the predicted digit on the canvas
            font = pygame.font.Font(None, 120)
            text = font.render(str(digit), True, WHITE)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            canvas.blit(text, text_rect)

    # Update the window
    WIN.fill(BLACK)
    WIN.blit(canvas, (0, 0))
    pygame.display.flip()

# Quit the application
pygame.quit()

