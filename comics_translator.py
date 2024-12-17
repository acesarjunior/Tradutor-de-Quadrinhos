import fitz  # PyMuPDF
import os
from ultralytics import YOLO
import easyocr
import language_tool_python
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import re
import requests
import time

# Configurações
temp_dir = "temp"
translated_dir = "translated_books"
font_path = "arial.ttf"
initial_font_size = 300  # Tamanho inicial da fonte
font_size_multiplier = 5  # Multiplicador do tamanho da fonte
yolo_model_path = 'modelo/runs/detect/ballon_model/weights/best.pt'
local_opus_dir = "modelo/opus-mt-tc-big-en-pt"  # Diretório para armazenar o modelo de tradução

# Flag para controlar se o PDF deve ser convertido em imagens
convert_pdf_to_images_flag = False  # Converte PDF em imagens, se necessário

# Inicialização
reader = easyocr.Reader(['en'], gpu=True)
tool = language_tool_python.LanguageTool('en-US')
tool2 = language_tool_python.LanguageTool('pt-BR')

# Verificar se CUDA (GPU) está disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Inicialização do modelo de tradução
translation_model, tokenizer = None, None


def download_file(url, save_path):
    """
    Downloads a file from a URL with a progress bar and saves it locally.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        downloaded_size = 0
        start_time = time.time()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)

                    # Calculate progress
                    elapsed_time = time.time() - start_time
                    speed = downloaded_size / 1024 / elapsed_time if elapsed_time > 0 else 0
                    progress = downloaded_size / total_size * 100 if total_size > 0 else 0

                    # Display progress bar
                    print(f"\rDownloading {os.path.basename(save_path)}: [{progress:.2f}%] "
                          f"{downloaded_size / 1024:.2f} KB of {total_size / 1024:.2f} KB at {speed:.2f} KB/s", end="")

        print()  # Move to next line after completion


def download_and_store_mbart():
    """
    Downloads the Opus MT TC-BIG model files directly from Hugging Face and saves them locally.
    """
    if not os.path.exists(local_opus_dir):
        os.makedirs(local_opus_dir, exist_ok=True)

    print("Downloading Opus MT TC-BIG model files from Hugging Face...")
    base_url = "https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt/resolve/main"
    files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "source.spm",
        "target.spm",
        "vocab.json",  # Arquivo necessário
        "special_tokens_map.json",
    ]

    for file_name in files:
        file_url = f"{base_url}/{file_name}"
        save_path = os.path.join(local_opus_dir, file_name)
        if not os.path.exists(save_path):  # Download only if the file is not already present
            download_file(file_url, save_path)

    print(f"All Opus MT TC-BIG model files have been downloaded to: {local_opus_dir}")


def init_translation_model():
    """
    Inicializa o modelo Opus MT TC-BIG a partir do diretório local.
    """
    global translation_model, tokenizer
    translation_model = MarianMTModel.from_pretrained(local_opus_dir).to(device)
    tokenizer = MarianTokenizer.from_pretrained(local_opus_dir)
    print("Modelo de tradução carregado com sucesso.")


def convert_pdf_to_images(pdf_path):
    """
    Converte PDF em imagens para processamento posterior.
    """
    if convert_pdf_to_images_flag:
        print(f"Converting PDF to images: {pdf_path}")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap(dpi=300)
                image_path = f"{temp_dir}/page_{page_number + 1}.jpg"
                pix.save(image_path)
                image_paths.append(image_path)
                print(f"Page {page_number + 1} - Done!")
            print(f"Converted {len(image_paths)} pages.")
            return image_paths
        except Exception as e:
            print(f"Error during PDF conversion: {e}")
            return []
    else:
        print("Skipping PDF to image conversion due to the flag being False.")
        image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.jpg')]
        if not image_paths:
            print("No images found in the directory. The program will stop.")
            return []
        else:
            print(f"Found {len(image_paths)} images in the directory. Continuing with these images.")
            return image_paths


def sort_images_by_page_number(image_paths):
    """
    Ordena as imagens numericamente com base no número da página.
    """
    def extract_page_number(filename):
        match = re.search(r'page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    return sorted(image_paths, key=extract_page_number)


def detect_and_translate_images(image_paths, yolo_model_path, conf_threshold=0.25):
    """
    Detecta e traduz texto em imagens usando YOLO e Opus MT TC-BIG.
    """
    print("Carregando o modelo YOLOv8...")
    yolo_model = YOLO(yolo_model_path)
    translated_images = []

    image_paths = sort_images_by_page_number(image_paths)  # Ordena corretamente as imagens

    for image_path in image_paths:
        print(f"Processando a imagem: {image_path}")
        try:
            results = yolo_model(image_path, conf=conf_threshold)
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            detections = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i, (x1, y1, x2, y2) in enumerate(detections):
                if confs[i] >= conf_threshold:
                    cropped_balloon = image.crop((x1, y1, x2, y2))
                    cropped_balloon_np = np.array(cropped_balloon)

                    results = reader.readtext(cropped_balloon_np, detail=0, contrast_ths=0.4, adjust_contrast=0.7)
                    text = " ".join(results)

                    if not text.strip():
                        print("Nenhum texto detectado.")
                        continue

                    corrected_text = text.capitalize()
                    corrected_text = tool.correct(corrected_text)
                    inputs = tokenizer(corrected_text, return_tensors="pt").to(device)
                    generated_tokens = translation_model.generate(**inputs)
                    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    pattern = r"\d+"  # Padrão para identificar números inteiros
                    translated_text = ' ' if re.fullmatch(pattern, translated_text) else translated_text
                    translated_text = re.sub(r'\d+', '', translated_text)
                    # Remove caracteres que não sejam letras com acentos, ç ou pontuações
                    translated_text = re.sub(r"[^a-zA-Zá-úÁ-ÚçÇ.,!?'\s\"]", '', translated_text)
                    # Remove consoantes isoladas
                    translated_text = re.sub(r'\b[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]\b', '', translated_text)
                    translated_text = tool2.correct(translated_text)
                    print(f"Texto extraído: {text}")
                    print(f"Texto traduzido: {translated_text}")

                    # Máscara branca circular
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    radius = max((x2 - x1), (y2 - y1)) / 2
                    draw.ellipse(
                        [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
                        fill="white"
                    )

                    # Inserir texto ajustado ao círculo
                    draw_text_in_mask(draw, translated_text, center_x, center_y, radius, font_path)

            translated_image_name = os.path.basename(image_path).replace(".jpg", "_translated.jpg")
            translated_image_path = os.path.join(temp_dir, translated_image_name)
            image.save(translated_image_path)
            translated_images.append(translated_image_path)
            print(f"Imagem traduzida salva em: {translated_image_path}")

        except Exception as e:
            print(f"Erro ao processar a imagem {image_path}: {e}")

    return translated_images


def draw_text_in_mask(draw, text, x, y, radius, font_path):
    """
    Adjusts the text to fit within a circular mask, wrapping and resizing as needed.
    """
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)
    max_width = 2 * radius
    max_height = 2 * radius

    # Function to calculate multi-line text size
    def multiline_textsize(lines, font):
        """Calculate the size of multi-line text."""
        width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
        return width, height

    # Reduce font size until the text fits within the mask
    while True:
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        text_width, text_height = multiline_textsize(lines, font)
        if text_width <= max_width and text_height <= max_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # Center and draw the text
    total_text_height = text_height
    start_y = y - radius + (max_height - total_text_height) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        draw.text((x - line_width / 2, start_y), line, font=font, fill="black")
        start_y += bbox[3] - bbox[1]


from PIL import Image
import os


from PIL import Image
import os


def convert_images_to_pdf(image_paths, output_pdf_path, dpi=150, quality=60):
    if not image_paths:
        print("Nenhuma imagem traduzida para converter em PDF.")
        return

    # Sort image paths numerically
    sorted_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    def image_generator(paths):
        for path in paths:
            with Image.open(path) as img:
                yield img.convert("RGB")

    # Create an iterator for images
    image_iter = image_generator(sorted_paths)

    # Save the PDF using the first image and appending the rest incrementally
    first_image = next(image_iter)
    first_image.save(
        output_pdf_path,
        save_all=True,
        append_images=list(image_iter),
        dpi=(dpi, dpi),
        quality=quality
    )
    print(f"PDF gerado em: {output_pdf_path}")


# Caminhos
pdf_filename = "buddha_v01.pdf"
pdf_path = os.path.join("books", pdf_filename)
output_pdf_path = os.path.join(translated_dir, pdf_filename.replace(".pdf", "_translated.pdf"))

# Executar o programa
download_and_store_mbart()
init_translation_model()

image_paths = convert_pdf_to_images(pdf_path)

# Check if translated images already exist
translated_image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('_translated.jpg')]

if translated_image_paths:
    print("Translated images already exist. Skipping translation and converting to PDF.")
    convert_images_to_pdf(translated_image_paths, output_pdf_path)
else:
    # Perform translation and then convert to PDF
    translated_images = detect_and_translate_images(image_paths, yolo_model_path)
    convert_images_to_pdf(translated_images, output_pdf_path)

