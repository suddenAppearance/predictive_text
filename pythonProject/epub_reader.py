import ebooklib
from ebooklib import epub

book = epub.read_epub('tolstoy.epub')
for image in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    print(image.get_content())