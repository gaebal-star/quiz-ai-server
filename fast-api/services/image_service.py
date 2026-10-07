import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from common import MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE

def normalize_image(data: bytes, declared_mime: str) -> tuple[bytes, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}.get(image.format)
                if mime is None or (declared_mime and declared_mime != mime):
                    raise ValueError('JPG, PNG, WebP 형식과 MIME 타입을 확인해주세요.')
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError('이미지는 2500만 픽셀 이하여야 합니다.')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('움직이는 이미지 대신 정지 이미지를 사용해주세요.')
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image = ImageOps.exif_transpose(image)
                if max(image.size) > MAX_IMAGE_SIDE:
                    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                # 투명 배경을 흰색으로 합성. 작은 글자 보존을 위해 무손실 PNG 사용.
                rgba = image.convert('RGBA')
                background = Image.new('RGBA', rgba.size, 'white')
                background.alpha_composite(rgba)
                output = io.BytesIO()
                background.convert('RGB').save(output, format='PNG')
                return output.getvalue(), 'image/png'
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('손상되었거나 해상도가 너무 큰 이미지입니다.') from exc
