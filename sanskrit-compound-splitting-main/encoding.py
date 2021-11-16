import sys
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

unicode_iast_map = {
    "्": "",
    "ॐ": "oṁ",
    "ँ": "aṁ",
    "ं": "aṃ",
    "ः": "aḥ",
    "ख": "kh",
    "क": "k",
    "घ": "gh",
    "ग": "g",
    "ङ": "ṅ",
    "छ": "ch",
    "च": "c",
    "झ": "jh",
    "ज": "j",
    "ञ": "ñ",
    "ठ": "ṭh",
    "ट": "ṭ",
    "ढ": "ḍh",
    "ड": "ḍ",
    "ण": "ṇ",
    "थ": "th",
    "त": "t",
    "ध": "dh",
    "द": "d",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "भ": "bh",
    "ब": "b",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "ś",
    "ष": "ṣ",
    "स": "s",
    "ह": "h",
    "ै": "ai",
    "ौ": "au",
    "": "a",
    "ा": "ā",
    "ि": "i",
    "ी": "ī",
    "ु": "u",
    "ू": "ū",
    "ृ": "ṛ",
    "ॄ": "ṝ",
    "ॢ": "ḷ",
    "े": "e",
    "ो": "o",
    "ऽ": "'",
    "ऐ": "ai",
    "औ": "au",
    "अ": "a",
    "आ": "ā",
    "इ": "i",
    "ई": "ī",
    "उ": "u",
    "ऊ": "ū",
    "ऋ": "ṛ",
    "ॠ": "ṝ",
    "ऌ": "ḷ",
    "ॡ": "ḹ",
    "ए": "e",
    "ओ": "o",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
    "०": "0",
    "।": ".",
}

vowels = [
    "्",
    "ै",
    "ौ",
    "",
    "ा",
    "ि",
    "ी",
    "ु",
    "ू",
    "ृ",
    "ॄ",
    "ॢ",
    "े",
    "ो",
    "ँ",
    "ं",
    "ः",
    "्",
    "अ",
    "आ",
    "इ",
    "ई",
    "उ",
    "ऊ",
    "ऋ",
    "ऌ",
    "ए",
    "ऐ",
    "ओ",
    "औ",
    "ॐ",
    "ॠ",
    "ॡ",
]

iast_internal_map = [
    ("ā", "A"),
    ("ī", "I"),
    ("ū", "U"),
    ("ṛ", "R"),
    ("ṝ", "L"),
    ("ai", "E"),
    ("au", "O"),
    ("kh", "K"),
    ("gh", "G"),
    ("ṅ", "F"),
    ("ch", "C"),
    ("jh", "J"),
    ("ñ", "Q"),
    ("ṭh", "W"),
    ("ṭ", "w"),
    ("ḍh", "X"),
    ("ḍ", "x"),
    ("ṇ", "N"),
    ("th", "T"),
    ("dh", "D"),
    ("ph", "P"),
    ("bh", "B"),
    ("ś", "S"),
    ("ṣ", "z"),
    ("ṃ", "M"),
    ("ḥ", "H"),
    (" ", "_"),
]


def old_unicode_to_iast(text):
    iast_text = ""
    is_vowel = True

    for x in list(text):
        if x not in unicode_iast_map:
            iast_text += x
            continue
        if x == " ":
            if not is_vowel:
                iast_text += "a"
            iast_text += " "
            is_vowel = True
            continue
        if not is_vowel and x not in vowels:
            iast_text += "a"
        iast_text += unicode_iast_map[x]
        is_vowel = x in vowels

    return iast_text


def iast_to_unicode(text):
    return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)

def unicode_to_iast(text):
    return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)

def iast_to_internal(text):
    for src, dst in iast_internal_map:
        text = text.replace(src, dst)
    return text

def internal_to_iast(text):
    for src, dst in iast_internal_map:
        text = text.replace(dst, src)
    return text

def unicode_to_internal(text):
    text = unicode_to_iast(text)
    text = iast_to_internal(text)
    return text

def internal_to_unicode(text):
    text = internal_to_iast(text)
    text = iast_to_unicode(text)
    return text


if __name__ == "__main__":

    text = "प्रस्तुतोऽयं पाठः श्रीपद्मशास्त्रिणा विरचितम् 'विश्वकथाशतकम्' इति कथासङ्ग्रहात् गृहीतोऽस्ति।"

    iast_text = unicode_to_iast(text)
    print(iast_text)
    internal_text = iast_to_internal(iast_text)
    print(internal_text)
