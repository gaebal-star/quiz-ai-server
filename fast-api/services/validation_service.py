import re
import unicodedata

def normalize_text(value: str) -> str:
    # 코드/수식 줄바꿈을 보존하고 눈에 보이지 않는 문자만 제거한다.
    return unicodedata.normalize('NFC', value).replace('\u200b', '').replace('\ufeff', '').strip()

def signature(value):
    return re.sub(r'\s+', '', normalize_text(value).casefold())

def normalize_study_material(material):
    ids = set()
    for chunk in material.chunks:
        if chunk.chunkId in ids:
            raise ValueError('중복 청크 ID')
        ids.add(chunk.chunkId)
        chunk.topic = normalize_text(chunk.topic)
        chunk.content = normalize_text(chunk.content)
        chunk.keyPoints = [normalize_text(x) for x in chunk.keyPoints if normalize_text(x)]
        chunk.evidence = [normalize_text(x) for x in chunk.evidence if normalize_text(x)]
        if not chunk.topic or not chunk.content or not chunk.evidence:
            raise ValueError('자료 내용 또는 출처 근거가 비어 있습니다.')
    return material

def validate_quizzes(response, quiz_type, count, material):
    if len(response.quizzes) != count:
        raise ValueError(f'문항 수는 정확히 {count}개여야 합니다.')
    ids = {c.chunkId for c in material.chunks}
    seen = set()
    for quiz in response.quizzes:
        for name in ('quizQuestion', 'quizCorrectAnswer', 'quizExplanation'):
            value = normalize_text(getattr(quiz, name))
            if not value:
                raise ValueError('문제/정답/해설이 비어 있습니다.')
            setattr(quiz, name, value)
        quiz.quizSelections = [normalize_text(x) for x in quiz.quizSelections]
        if quiz.sourceChunkId not in ids:
            raise ValueError('정답 근거 청크 ID가 존재하지 않습니다.')
        key = signature(quiz.quizQuestion)
        if key in seen:
            raise ValueError('중복 문제입니다.')
        seen.add(key)
        choices = quiz.quizSelections
        if any(not c for c in choices):
            raise ValueError('빈 보기입니다.')
        if quiz_type == 'MULTIPLE_CHOICE':
            if len(choices) != 4 or len({signature(c) for c in choices}) != 4 or quiz.quizCorrectAnswer not in choices:
                raise ValueError('객관식은 서로 다른 보기 4개와 보기 문자열 정답이 필요합니다.')
        elif quiz_type == 'OX':
            if choices != ['O', 'X'] or quiz.quizCorrectAnswer not in choices:
                raise ValueError('OX 보기는 [O, X], 정답은 O 또는 X여야 합니다.')
        elif quiz_type == 'SHORT_ANSWER':
            if choices:
                raise ValueError('단답형 보기는 빈 배열이어야 합니다.')
        else:
            raise ValueError('지원하지 않는 문제 유형입니다.')
    return response
