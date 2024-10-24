from PIL import Image, ImageDraw
import uuid

square_size = 20
image_size = (8 * square_size, 8 * square_size)

def draw(grid):
    img = Image.new('RGB', image_size, "white")
    draw = ImageDraw.Draw(img)

    for row in range(8):
        for col in range(8):
            if grid[row][col] == 1:
                # Draw a filled square
                top_left = (col * square_size, row * square_size)
                bottom_right = ((col + 1) * square_size, (row + 1) * square_size)
                draw.rectangle([top_left, bottom_right], fill="black")
            if grid[row][col] == 2:
                # Draw a filled square
                top_left = (col * square_size, row * square_size)
                bottom_right = ((col + 1) * square_size, (row + 1) * square_size)
                draw.rectangle([top_left, bottom_right], fill="red")

    img.save(f"imgs/image_{uuid.uuid4()}.png")