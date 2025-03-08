import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def letterbox_image(image, size, fill_color=(0, 0, 0)):
    """
    Redimensiona a imagem para que o maior lado se ajuste ao tamanho 'size'
    e adiciona padding (letterbox) para que a imagem final seja um quadrado de dimensão size x size.
    
    Args:
        image (PIL.Image): Imagem de entrada.
        size (int): Tamanho desejado para a dimensão final (quadrada).
        fill_color (tuple): Cor de preenchimento para as bordas (default: preto).
    
    Returns:
        PIL.Image: Imagem resultante com letterbox.
    """
    iw, ih = image.size
    scale = size / max(iw, ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (size, size), fill_color)
    pad_left = (size - nw) // 2
    pad_top = (size - nh) // 2
    new_image.paste(image_resized, (pad_left, pad_top))
    return new_image

def main():
    parser = argparse.ArgumentParser(
        description="Redimensiona uma imagem usando letterbox (zoom out) para preservar todo o conteúdo e a relação de aspecto, e salva o resultado."
    )
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem original")
    parser.add_argument("--size", type=int, default=224, help="Tamanho de saída desejado (ex.: 224 ou 112)")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    final_image = letterbox_image(image, args.size)
    
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_image_path = f"{base_name}_resized_{args.size}x{args.size}.png"
    final_image.save(output_image_path)
    print(f"Imagem salva em: {output_image_path}")

if __name__ == "__main__":
    main()
