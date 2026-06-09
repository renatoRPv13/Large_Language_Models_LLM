import os
import re
import requests
import uuid
from urllib.parse import quote
from fastapi import HTTPException, Depends
from sqlmodel import Session, select

import sys
import os

# Adicionar o caminho raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora as importações devem funcionar
#from GenAnkiCards.task3_image_and_audio.schemas import logger, AudioRequest, get_session, FlashcardDB, slugify

from GenAnkiCards.task3_image_and_audio.schemas import ImageRequest, get_session, FlashcardDB, logger, slugify

# Images directory setup # Configuração do diretório de imagens
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "./images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Dummy image paths # Caminhos de imagem fictícios
DUMMY_IMAGE_FS_PATH = os.path.join(IMAGES_DIR, "dummy.png")
DUMMY_IMAGE_URL = "./images/dummy.png"


def generate_image(req: ImageRequest, session: Session = Depends(get_session)):
    logger.info(
        f"Iniciando a geração de imagens para a palavra: '{req.word}', language: '{req.language}', flashcard_id: {req.flashcard_id}"
    )

    # Check if the image already exists in the database for this word/language combination
    # Verificar se a imagem já existe no banco de dados para esta combinação de palavra/idioma
    if req.flashcard_id:
        card = session.get(FlashcardDB, req.flashcard_id)
        if card and card.image_path:
            logger.info(f"Flashcard {req.flashcard_id} already has an image: {card.image_path}")
            return {"image_path": card.image_path, "prompt": f"Existing image for '{req.word}'"}
    else:
        stmt = select(FlashcardDB).where(
            FlashcardDB.word == req.word,
            FlashcardDB.language == req.language,
            FlashcardDB.image_path.isnot(None)
        ).limit(1)
        existing_card_with_image = session.exec(stmt).first()

        if existing_card_with_image:
            logger.info(
                f"Found existing image for word '{req.word}' in language '{req.language}': "
                f"{existing_card_with_image.image_path}"
            )
            return {"image_path": existing_card_with_image.image_path, "prompt": f"Existing image for '{req.word}'"}

    # No existing image found, proceed with generation
    # Nenhuma imagem existente encontrada, prossiga com a geração
    prompt = (
        f"Crie uma ilustração completa, em tela cheia, da palavra '{req.word}' "
        f"no contexto do linguagem '{req.language}' language. "
        f"Certifique-se de que todo o assunto esteja visível dentro da moldura, com margens e espaçamento adequado. "
        f"Não corte nenhuma parte importante. Centralize o assunto principal com espaço em branco ou fundo suficiente ao redo. "
        f"Faça uma ilustração clara e didática, adequada para o aprendizado de vocabulário."
    )
    logger.info(f"prompt gerado: {prompt}")

    safe_word = slugify(req.word, "image")
    unique_id = uuid.uuid4().hex[:12]
    output_format = "png"
    filename = f"{safe_word}_{unique_id}.{output_format}"
    image_fs_path = os.path.join(IMAGES_DIR, filename)
    image_url_path = f"./images/{filename}"

    try:
        # Use Pollinations.ai (free, no API key required) # Use Pollinations.ai (gratuito, sem necessidade de chave de API)
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        logger.info(f"Chamando a Pollinations.ai para geração de imagens.: {url}")

        response = requests.get(url, timeout=120)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(image_fs_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Imagem salva no sistema de arquivos local: {image_fs_path}")

    except Exception as e:
        logger.exception(f"Image generation failed, falling back to dummy image. Error: {e}")
        # Ensure dummy file exists before trying to use it
        # Certifique-se de que o arquivo fictício exista antes de tentar usá-lo
        if not os.path.exists(DUMMY_IMAGE_FS_PATH):
            # Create a simple dummy png if it doesn't exist
            # Crie um png fictício simples caso ele não exista.
            try:
                from PIL import Image
                dummy_img = Image.new('RGB', (100, 100), color = 'red')
                dummy_img.save(DUMMY_IMAGE_FS_PATH, 'PNG')
            except Exception as dummy_e:
                logger.error(f"Could not create dummy image: {dummy_e}")

        image_url_path = DUMMY_IMAGE_URL

    # Persist to DB if flashcard_id provided
    # Persistir no banco de dados se o flashcard_id for fornecido
    if req.flashcard_id is not None:
        card = session.get(FlashcardDB, req.flashcard_id)
        if not card:
            raise HTTPException(status_code=404, detail="Flashcard not found for image update")

        card.image_path = image_url_path
        session.add(card)
        session.commit()
        session.refresh(card)

    logger.info(f"Processo de geração de imagens concluído. URL: {image_url_path}")
    return {"image_path": image_url_path, "prompt": prompt}
