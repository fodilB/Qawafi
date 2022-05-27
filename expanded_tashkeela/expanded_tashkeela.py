# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Arabic Vocalized Words Dataset."""


import glob
import os
import re 

import datasets
from diacritization_evaluation import util 

HARAQAT = ['ْ', 'ّ', 'ٌ', 'ٍ', 'ِ', 'ً', 'َ', 'ُ']
PUNCTUATIONS = ['.', '،', ':', '؛', '-', '؟']
ARAB_CHARS = 'ىعظحرسيشضق ثلصطكآماإهزءأفؤغجئدةخوبذتن'
ARAB_CHARS_NO_SPACE = 'ىعظحرسيشضقثلصطكآماإهزءأفؤغجئدةخوبذتن'
ARAB_CHARS_PUNCTUATIONS = ARAB_CHARS + ''.join(PUNCTUATIONS)
# VALID_ARABIC = HARAQAT + list(ARAB_CHARS)
BASIC_HARAQAT = {
    'َ': 'Fatha              ',
    'ً': 'Fathatah           ',
    'ُ': 'Damma              ',
    'ٌ': 'Dammatan           ',
    'ِ': 'Kasra              ',
    'ٍ': 'Kasratan           ',
    'ْ': 'Sukun              ',
    'ّ': 'Shaddah            ',
}
ALL_POSSIBLE_HARAQAT = {'': 'No Diacritic       ',
                        'َ': 'Fatha              ',
                        'ً': 'Fathatah           ',
                        'ُ': 'Damma              ',
                        'ٌ': 'Dammatan           ',
                        'ِ': 'Kasra              ',
                        'ٍ': 'Kasratan           ',
                        'ْ': 'Sukun              ',
                        'ّ': 'Shaddah            ',
                        'َّ': 'Shaddah + Fatha    ',
                        'ًّ': 'Shaddah + Fathatah ',
                        'ُّ': 'Shaddah + Damma    ',
                        'ٌّ': 'Shaddah + Dammatan ',
                        'ِّ': 'Shaddah + Kasra    ',
                        'ٍّ': 'Shaddah + Kasratan '}

VALID_ARABIC = list(ALL_POSSIBLE_HARAQAT.keys())+list(ARAB_CHARS_PUNCTUATIONS)

_DESCRIPTION = """\
Arabic vocalized texts.
it contains 75 million of fully vocalized words mainly\
97 books from classical and modern Arabic language.
"""

_CITATION = """\
@article{zerrouki2017tashkeela,
  title={Tashkeela: Novel corpus of Arabic vocalized texts, data for auto-diacritization systems},
  author={Zerrouki, Taha and Balla, Amar},
  journal={Data in brief},
  volume={11},
  pages={147},
  year={2017},
  publisher={Elsevier}
}
"""

_HOMEPAGE = "https://sourceforge.net/projects/tashkeela/"

_LICENSE = "GPLv2"

_DOWNLOAD_URL = "https://sourceforge.net/projects/tashkeela/files/latest/download"


class TashkeelaConfig(datasets.BuilderConfig):
    """BuilderConfig for Tashkeela."""

    def __init__(self, **kwargs):
        """BuilderConfig for Tashkeela.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TashkeelaConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Tashkeela(datasets.GeneratorBasedBuilder):
    """Tashkeela dataset."""

    BUILDER_CONFIGS = [
        TashkeelaConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "directory": os.path.join(arch_path, "Tashkeela-arabic-diacritized-text-utf8-0.3", "texts.txt")
                },
            ),
        ]

    def preprpocess(self, book):
      out = ""
      i = 0 
      while( i < len(book)):
        if i < len(book):
          if book[i] in BASIC_HARAQAT and book[i+1] in BASIC_HARAQAT:
            i += 1
            continue
        out += book[i]
        i += 1
        
      out = re.sub(' +', ' ', out) 
      out = re.sub("\n", '', out)
      return out
        
      book = re.sub(' +', ' ', out) 
      book = re.sub("\n", '', out)
      return book

    def chunkify(self, book, max_length=500):
      tokens = self.preprpocess(book)
      out = []
      for i in range(len(tokens) // max_length):
        text = tokens[i*max_length: (i+1)*max_length]
        try:
          util.extract_haraqat(text)
        except:
          print(text)
          continue
        out.append(text) 
      return out
    def _generate_examples(self, directory):
        """Generate examples."""
        cnt = 0 
        for id_, file_name in enumerate(sorted(glob.glob(os.path.join(directory, "**.txt")))):
            with open(file_name, encoding="UTF-8") as f:
                for chunk in self.chunkify(f.read()):
                  cnt += 1
                  yield str(cnt), {"text": chunk}
